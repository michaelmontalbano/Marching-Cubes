#!/usr/bin/env python3
"""
Train Flow Matching model for MESH prediction

Trains a conditional flow matching model to predict MESH at discrete timesteps.
- Input: (12, 256, 256, 8) - 12 timesteps with 8 radar/weather channels (condition)
- Target: (256, 256, 1) - Single timestep MESH prediction
- Training: Randomly samples one of the 12 target timesteps per batch
- Uses min-max normalization with global parameters
"""

import os
import sys
import json
import gc
import numpy as np
import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# Import custom model
from flow_matching_model import FlowMatchingUNet

# Import custom losses if available
try:
    from models import weighted_mse, csi
except:
    # Fallback if models.py not available
    def weighted_mse(weight=2.0):
        def loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            return mse
        return loss
    
    def csi(y_true, y_pred, threshold=25.4):
        y_true_binary = tf.cast(y_true > threshold, tf.float32)
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        
        hits = tf.reduce_sum(y_true_binary * y_pred_binary)
        false_alarms = tf.reduce_sum(y_pred_binary * (1 - y_true_binary))
        misses = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
        
        csi_score = hits / (hits + false_alarms + misses + 1e-7)
        return csi_score

# ---------------- Configuration ----------------
BUCKET = 'dev-grib-bucket'
TRAIN_DF_PATH = 'dataframes/train.csv'
VAL_DF_PATH = 'dataframes/val.csv'
TEST_DF_PATH = 'dataframes/test.csv'
NORM_MIN_PATH = 'global_mins.npy'
NORM_MAX_PATH = 'global_maxs.npy'

# Model configuration
BATCH_SIZE = 8
EPOCHS = 100
INITIAL_FILTERS = 64
LEARNING_RATE = 1e-4
KERNEL_SIZE = 3
NUM_DOWNSAMPLING = 4
DROPOUT_RATE = 0.1

# Data shape
TIMESTEPS = 12
HEIGHT = 256
WIDTH = 256
INPUT_CHANNELS = 8
OUTPUT_CHANNELS = 1

# Channels to normalize (all except binary mask at index 7)
NORM_CHANNELS = [0, 1, 2, 3, 4, 5, 6]

# Flow matching parameters
SIGMA_MIN = 1e-4  # Minimum noise level

# ---------------- Data Loader ----------------
class DataLoader:
    def __init__(self, bucket_name, s3_client):
        self.bucket_name = bucket_name
        self.s3_client = s3_client
        self.global_min = None
        self.global_max = None
        
    def load_dataframe_from_s3(self, file_key):
        """Loads dataframe from S3 CSV file."""
        s3_path = f's3://{self.bucket_name}/{file_key}'
        df = pd.read_csv(s3_path)
        return df

    def load_npy_from_s3(self, key):
        """Loads .npy file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = np.load(BytesIO(response['Body'].read()))
            return data
        except Exception as e:
            print(f'Error loading s3://{self.bucket_name}/{key}: {e}')
            return None
    
    def load_normalization_params(self):
        """Load global min/max normalization parameters."""
        self.global_min = self.load_npy_from_s3(NORM_MIN_PATH)
        self.global_max = self.load_npy_from_s3(NORM_MAX_PATH)
        
        if self.global_min is None or self.global_max is None:
            raise ValueError("Failed to load normalization parameters")
        
        print(f"Loaded normalization params: min shape={self.global_min.shape}, max shape={self.global_max.shape}")
        return True

    def normalize_field(self, data):
        """
        Normalizes data using global min/max to [0,1] range.
        
        Args:
            data: Array of shape (timesteps, height, width, channels)
        
        Returns:
            Normalized array
        """
        if len(data.shape) < 4:
            data = np.expand_dims(data, axis=-1)
        
        normalized = np.zeros_like(data, dtype=np.float32)
        
        for t in range(data.shape[0]):
            for c in range(data.shape[-1]):
                if c in NORM_CHANNELS:
                    # Min-max normalization
                    min_val = self.global_min[c]
                    max_val = self.global_max[c]
                    normalized[t, :, :, c] = (data[t, :, :, c] - min_val) / (max_val - min_val + 1e-5)
                    normalized[t, :, :, c] = np.where(normalized[t, :, :, c] <= 0, 1e-5, normalized[t, :, :, c])
                else:
                    # Keep mask as-is (channel 7)
                    normalized[t, :, :, c] = data[t, :, :, c]
        
        return normalized

# ---------------- Input Pipeline ----------------
class FlowMatchingInputPipeline:
    def __init__(self, batch_size, data_loader, train_df):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.train_df = train_df
        self.s3_client = boto3.client('s3')
        
    def data_generator(self):
        """Generator for flow matching training data."""
        while True:
            # Randomly sample batch. Allow sampling with replacement when the
            # requested batch size exceeds the dataset size (e.g. when the
            # global batch size is scaled for multi-GPU training).
            replace = self.batch_size > len(self.train_df)
            selected_rows = self.train_df.sample(self.batch_size, replace=replace)
            
            batch_conditions = []  # Input context (all 12 timesteps)
            batch_targets = []     # Target MESH (single timestep)
            batch_noise = []       # Noise samples
            batch_t = []           # Flow times [0, 1]
            batch_timestep_idx = []  # Which of 12 timesteps we're predicting
            
            for idx, row in selected_rows.iterrows():
                # Get input and target paths
                input_path = f"data/int5/input{row.file_path.split('input')[1]}"
                target_path = f'data/int5/mesh_swath_intervals{row.file_path.split("input")[1]}'
                
                # Load data
                inputs = self.data_loader.load_npy_from_s3(input_path)
                targets = self.data_loader.load_npy_from_s3(target_path)
                
                if inputs is None or targets is None:
                    print(f"Skipping invalid file: {input_path}")
                    continue
                
                # Expand dims if needed
                if inputs.ndim < 4:
                    inputs = np.expand_dims(inputs, axis=-1)
                
                # Reverse time (oldest to newest)
                inputs = inputs[::-1, :, :, :]
                
                # Take first 12 timesteps and 8 channels
                inputs = inputs[:TIMESTEPS, :, :, :INPUT_CHANNELS]
                
                # Normalize inputs
                inputs = self.data_loader.normalize_field(inputs)
                inputs[np.isnan(inputs)] = 0.0
                
                # Process targets
                if len(targets.shape) < 4:
                    targets = np.expand_dims(targets, axis=-1)
                
                # Take first 12 timesteps, channel 0 (MESH)
                targets = targets[:TIMESTEPS, :, :, 0:1]
                targets[np.isnan(targets)] = 0.0
                
                # Randomly select one timestep to predict (0-11)
                timestep_idx = np.random.randint(0, TIMESTEPS)
                target_t = targets[timestep_idx]  # Shape: (256, 256, 1)
                
                # Sample flow time uniformly from [0, 1]
                t = np.random.uniform(0, 1)
                
                # Sample noise from standard normal
                noise = np.random.randn(*target_t.shape).astype(np.float32)
                
                batch_conditions.append(inputs)
                batch_targets.append(target_t)
                batch_noise.append(noise)
                batch_t.append(t)
                batch_timestep_idx.append(timestep_idx)
            
            if len(batch_conditions) > 0:
                batch_conditions = np.array(batch_conditions)
                batch_targets = np.array(batch_targets)
                batch_noise = np.array(batch_noise)
                batch_t = np.array(batch_t).astype(np.float32)
                batch_timestep_idx = np.array(batch_timestep_idx).astype(np.float32)
                
                # Clean NaNs
                batch_conditions[np.isnan(batch_conditions)] = 0
                batch_targets[np.isnan(batch_targets)] = 0
                
                # Compute interpolated state: x_t = (1-t) * noise + t * target
                # Expand t for broadcasting: (batch, 1, 1, 1)
                t_expanded = batch_t[:, None, None, None]
                x_t = (1 - t_expanded) * batch_noise + t_expanded * batch_targets
                
                # Compute target velocity field: v = target - noise
                target_velocity = batch_targets - batch_noise
                
                yield (
                    {
                        'x_t': x_t,
                        'condition': batch_conditions,
                        't': batch_t,
                        'timestep_idx': batch_timestep_idx
                    },
                    target_velocity
                )

    def create_dataset(self):
        """Creates TensorFlow dataset from generator."""
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                {
                    'x_t': tf.TensorSpec(shape=(self.batch_size, HEIGHT, WIDTH, OUTPUT_CHANNELS), dtype=tf.float32),
                    'condition': tf.TensorSpec(shape=(self.batch_size, TIMESTEPS, HEIGHT, WIDTH, INPUT_CHANNELS), dtype=tf.float32),
                    't': tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32),
                    'timestep_idx': tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32)
                },
                tf.TensorSpec(shape=(self.batch_size, HEIGHT, WIDTH, OUTPUT_CHANNELS), dtype=tf.float32)
            )
        ).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset

# ---------------- Model Trainer ----------------
class ModelTrainer:
    def __init__(self, height, width, input_channels, timesteps,
                 initial_filters, num_downsampling, dropout_rate):
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.timesteps = timesteps
        self.initial_filters = initial_filters
        self.num_downsampling = num_downsampling
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """Build flow matching U-Net model."""
        unet = FlowMatchingUNet(
            height=self.height,
            width=self.width,
            input_channels=self.input_channels,
            timesteps=self.timesteps,
            initial_filters=self.initial_filters,
            num_downsampling=self.num_downsampling,
            dropout_rate=self.dropout_rate
        )
        self.model = unet.build_model()
        return self.model
    
    def compile_model(self, learning_rate):
        """Compile model with optimizer and loss."""
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            alpha=0.1
        )
        
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        
        # Simple MSE loss for velocity field prediction
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
    def train(self, train_dataset, val_dataset, epochs, steps_per_epoch, val_steps):
        """Train the model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = 'flow_matching'
        
        # Create directories
        os.makedirs(f'models/{version}', exist_ok=True)
        os.makedirs(f'logs/{version}', exist_ok=True)
        os.makedirs(f'checkpoints/{version}', exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f'checkpoints/{version}/best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=1,
                mode='min',
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=f'logs/{version}/{timestamp}',
                histogram_freq=5,
                write_graph=True,
                update_freq='batch'
            ),
            keras.callbacks.CSVLogger(f'logs/{version}/{timestamp}.csv')
        ]
        
        # Train
        print(f"Starting training: {epochs} epochs, {steps_per_epoch} steps/epoch")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            verbose=1,
            callbacks=callbacks
        )
        
        # Save final model
        model_name = f'flow_matching_time-{timestamp}_filters{self.initial_filters}.keras'
        self.model.save(f'models/{version}/{model_name}')
        print(f"Model saved: models/{version}/{model_name}")
        
        # Upload to S3
        s3 = boto3.client('s3')
        s3.upload_file(
            f'models/{version}/{model_name}',
            BUCKET,
            f'models/{version}/{model_name}'
        )
        print(f"Model uploaded to S3: s3://{BUCKET}/models/{version}/{model_name}")
        
        return history

# ---------------- Main ----------------
def main():
    print("="*80)
    print("TRAIN FLOW MATCHING U-NET FOR MESH PREDICTION")
    print("="*80)
    
    # Initialize
    s3_client = boto3.client('s3')
    data_loader = DataLoader(BUCKET, s3_client)
    
    # Load normalization params
    print("Loading normalization parameters...")
    data_loader.load_normalization_params()
    
    # Load dataframes
    print("Loading dataframes...")
    train_df = data_loader.load_dataframe_from_s3(TRAIN_DF_PATH)
    val_df = data_loader.load_dataframe_from_s3(VAL_DF_PATH)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Create datasets
    print("Creating flow matching datasets...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replica(s)")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using default TensorFlow strategy (no multi-GPU detected)")

    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")

    train_pipeline = FlowMatchingInputPipeline(global_batch_size, data_loader, train_df)
    val_pipeline = FlowMatchingInputPipeline(global_batch_size, data_loader, val_df)
    
    train_dataset = train_pipeline.create_dataset()
    val_dataset = val_pipeline.create_dataset()
    
    # Calculate steps
    steps_per_epoch = max(1, len(train_df) // global_batch_size)
    val_steps = max(1, len(val_df) // global_batch_size)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {val_steps}")
    
    # Build and compile model
    print("Building flow matching U-Net...")
    with strategy.scope():
        trainer = ModelTrainer(
            height=HEIGHT,
            width=WIDTH,
            input_channels=INPUT_CHANNELS,
            timesteps=TIMESTEPS,
            initial_filters=INITIAL_FILTERS,
            num_downsampling=NUM_DOWNSAMPLING,
            dropout_rate=DROPOUT_RATE
        )

        trainer.build_model()
        trainer.compile_model(LEARNING_RATE)
    
    # Train
    print("Training...")
    history = trainer.train(
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps
    )
    
    print("\nTraining complete!")
    print("To generate predictions, use Euler sampling with the trained model")

if __name__ == '__main__':
    main()