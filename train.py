#!/usr/bin/env python3
"""
Train ConvGRU model for MESH prediction

Trains a ConvGRU model to predict 12 timesteps of MESH from 8 input channels.
- Input: (12, 256, 256, 8) - 12 timesteps with 8 radar/weather channels
- Target: (12, 256, 256, 1) - 12 timesteps of MESH predictions
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

# Import custom layers from rnn.py and models.py
from rnn import (
    slice_to_n_steps, slice_output_shape,
    ResBlock, ConvGRU, ConvBlock,
    ZeroLikeLayer, ReflectionPadding2D, ResGRU, GRUResBlock, rnn
)
from models import weighted_mse, csi

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
INITIAL_FILTERS = 16
LEARNING_RATE = 3.68e-4
KERNEL_SIZE = 3
NUM_DOWNSAMPLING = 3
DROPOUT_RATE = 0.27

# Data shape
TIMESTEPS = 12
HEIGHT = 256
WIDTH = 256
INPUT_CHANNELS = 8
OUTPUT_CHANNELS = 1

# Channels to normalize (all except binary mask at index 7)
NORM_CHANNELS = [0, 1, 2, 3, 4, 5, 6]

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
class InputPipeline:
    def __init__(self, batch_size, data_loader, train_df):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.train_df = train_df
        self.s3_client = boto3.client('s3')
        
    def data_generator(self):
        """Generator for training data batches."""
        while True:
            # Randomly sample batch
            selected_rows = self.train_df.sample(self.batch_size)
            
            batch_inputs = []
            batch_labels = []
            
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
                
                batch_inputs.append(inputs)
                batch_labels.append(targets)
            
            if len(batch_inputs) > 0:
                batch_inputs = np.array(batch_inputs)
                batch_labels = np.array(batch_labels)
                
                # Clean NaNs
                batch_inputs[np.isnan(batch_inputs)] = 0
                batch_labels[np.isnan(batch_labels)] = 0
                
                yield batch_inputs, batch_labels

    def create_dataset(self):
        """Creates TensorFlow dataset from generator."""
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, TIMESTEPS, HEIGHT, WIDTH, INPUT_CHANNELS), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, TIMESTEPS, HEIGHT, WIDTH, OUTPUT_CHANNELS), dtype=tf.float32)
            )
        ).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset

# ---------------- Model Trainer ----------------
class ModelTrainer:
    def __init__(self, initial_filters, kernel_size, num_downsampling, dropout_rate):
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.num_downsampling = num_downsampling
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """Build RNN model using rnn() from rnn.py"""
        # The rnn() function signature from rnn.py:
        # rnn(timesteps, height, width, channels, other_fields_shape, 
        #     initial_filters, final_activation, dropout_rate, l1_reg, l2_reg, 
        #     x_pad, y_pad, kernel_size, padding, num_downsampling, future_channels)
        
        self.model = rnn(
            timesteps=TIMESTEPS,
            height=HEIGHT,
            width=WIDTH,
            channels=INPUT_CHANNELS,
            other_fields_shape=(TIMESTEPS, HEIGHT, WIDTH, INPUT_CHANNELS - 1),  # Not used but required
            initial_filters=self.initial_filters,
            final_activation='linear',  # Changed to linear since we're predicting continuous values
            dropout_rate=self.dropout_rate,
            l1_reg=0.28,
            l2_reg=0.29,
            x_pad=0,
            y_pad=0,
            kernel_size=self.kernel_size,
            padding='same',
            num_downsampling=self.num_downsampling,
            future_channels=INPUT_CHANNELS - 1  # 7 channels (excluding mask)
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        self.model.summary()
        return self.model
    
    def compile_model(self, learning_rate):
        """Compile model with optimizer and loss."""
        # Simple learning rate schedule - can be replaced with custom scheduler
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            alpha=0.1
        )
        
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=weighted_mse(),
            metrics=['accuracy', csi]
        )
        
    def train(self, train_dataset, val_dataset, epochs, steps_per_epoch, val_steps):
        """Train the model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = 'growth'
        
        # Create directories
        os.makedirs(f'models/{version}', exist_ok=True)
        os.makedirs(f'logs/{version}', exist_ok=True)
        os.makedirs(f'checkpoints/{version}', exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f'checkpoints/{version}/best_model.keras',
                monitor='val_csi',
                save_best_only=True,
                mode='max',
                verbose=1,
                initial_value_threshold=0.3
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_csi',
                patience=40,
                verbose=1,
                mode='max',
                restore_best_weights=True,
                start_from_epoch=40
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
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
        model_name = f'rnn_time-{timestamp}_version-{version}_filters{self.initial_filters}.keras'
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
    print("TRAIN CONVGRU MODEL FOR MESH PREDICTION")
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
    print("Creating datasets...")
    train_pipeline = InputPipeline(BATCH_SIZE, data_loader, train_df)
    val_pipeline = InputPipeline(BATCH_SIZE, data_loader, val_df)
    
    train_dataset = train_pipeline.create_dataset()
    val_dataset = val_pipeline.create_dataset()
    
    # Calculate steps
    steps_per_epoch = len(train_df) // BATCH_SIZE
    val_steps = len(val_df) // BATCH_SIZE
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {val_steps}")
    
    # Build and compile model
    print("Building model...")
    trainer = ModelTrainer(
        initial_filters=INITIAL_FILTERS,
        kernel_size=KERNEL_SIZE,
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
    
    # Print final metrics
    final_val_csi = history.history['val_csi'][-1]
    print(f"\nFinal validation CSI: {final_val_csi:.4f}")
    print("Training complete!")

if __name__ == '__main__':
    main()