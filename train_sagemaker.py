#!/usr/bin/env python3
"""
SageMaker training script for ConvGRU MESH prediction (aligned with train.py)

Fixes:
- Accept both --model-dir and --model_dir (SageMaker passes snake_case).
- Read train/val CSVs from the local SageMaker channels instead of s3:// URIs.
- Save the model to SM_MODEL_DIR (/opt/ml/model) so SageMaker uploads it.
- Still mirrors to output dirs and (optionally) uploads to S3 (like train.py).
"""

import os
import sys
import json
import argparse
import gc
from io import BytesIO
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# ---- imports from your repo ----
from rnn import (
    slice_to_n_steps, slice_output_shape,
    ResBlock, ConvGRU, ConvBlock,
    ZeroLikeLayer, ReflectionPadding2D, ResGRU, GRUResBlock, rnn
)
from models import weighted_mse, csi


# ---------------- Argument Parser ----------------
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--initial_filters', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3.68e-4)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_downsampling', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.27)
    parser.add_argument('--l1_reg', type=float, default=0.28)
    parser.add_argument('--l2_reg', type=float, default=0.29)

    # Data / model parameters
    parser.add_argument('--bucket', type=str, default='dev-grib-bucket')
    parser.add_argument('--timesteps', type=int, default=12)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--input_channels', type=int, default=8)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--version', type=str, default='growth')

    # Normalization artifacts (keys or filenames)
    parser.add_argument('--norm_min_key', type=str, default='global_mins.npy')
    parser.add_argument('--norm_max_key', type=str, default='global_maxs.npy')

    # Filenames inside the channels (SageMaker File mode)
    parser.add_argument('--train_df_name', type=str, default='train.csv')
    parser.add_argument('--val_df_name', type=str, default='val.csv')

    # SageMaker specific paths (local inside the container)
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--model_dir', dest='model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str,
                        default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    # Optional: number of GPUs passed by SM
    parser.add_argument('--num-gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', '0')))

    return parser.parse_args()


# ---------------- Data Loader ----------------
class DataLoader:
    def __init__(self, bucket_name, s3_client, local_train_dir, local_val_dir,
                 norm_min_key='global_mins.npy', norm_max_key='global_maxs.npy'):
        self.bucket_name = bucket_name
        self.s3_client = s3_client
        self.local_train_dir = local_train_dir
        self.local_val_dir = local_val_dir
        self.global_min = None
        self.global_max = None
        self.norm_min_key = norm_min_key
        self.norm_max_key = norm_max_key

    def _load_local_csv(self, dir_path, filename):
        path = os.path.join(dir_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    def load_train_df(self, filename):
        return self._load_local_csv(self.local_train_dir, filename)

    def load_val_df(self, filename):
        return self._load_local_csv(self.local_val_dir, filename)

    def load_npy_from_local_or_s3(self, key_or_filename):
        """
        Try local first (both train/val dirs), then S3 (bucket/key).
        This lets you stage mins/maxes in the train channel if you want.
        """
        local_candidates = [
            os.path.join(self.local_train_dir, os.path.basename(key_or_filename)),
            os.path.join(self.local_val_dir, os.path.basename(key_or_filename)),
        ]
        for candidate in local_candidates:
            if os.path.exists(candidate):
                try:
                    return np.load(candidate)
                except Exception as e:
                    print(f'Error loading local npy {candidate}: {e}')

        # Fallback to S3
        try:
            resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=key_or_filename)
            return np.load(BytesIO(resp['Body'].read()))
        except Exception as e:
            print(f'Error loading s3://{self.bucket_name}/{key_or_filename}: {e}')
            return None

    def load_normalization_params(self):
        self.global_min = self.load_npy_from_local_or_s3(self.norm_min_key)
        self.global_max = self.load_npy_from_local_or_s3(self.norm_max_key)
        if self.global_min is None or self.global_max is None:
            raise ValueError("Failed to load normalization parameters (mins/maxes)")
        print(f"Loaded normalization params: min shape={self.global_min.shape}, max shape={self.global_max.shape}")
        return True

    def normalize_field(self, data, norm_channels):
        """
        Min-max normalization to [0,1] for selected channels.
        Args:
            data: (T, H, W, C)
            norm_channels: indices of channels to normalize
        """
        if data.ndim < 4:
            data = np.expand_dims(data, axis=-1)

        normalized = np.zeros_like(data, dtype=np.float32)
        for t in range(data.shape[0]):
            for c in range(data.shape[-1]):
                if c in norm_channels:
                    mn = self.global_min[c]
                    mx = self.global_max[c]
                    val = (data[t, :, :, c] - mn) / (mx - mn + 1e-5)
                    normalized[t, :, :, c] = np.where(val <= 0, 1e-5, val)
                else:
                    normalized[t, :, :, c] = data[t, :, :, c]
        return normalized


# ---------------- Input Pipeline ----------------
class InputPipeline:
    def __init__(self, batch_size, data_loader, df, timesteps, height, width,
                 input_channels, output_channels, norm_channels):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.df = df
        self.timesteps = timesteps
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.norm_channels = norm_channels

    def data_generator(self):
        """
        Generator for training/validation batches.
        Assumes df has a column 'file_path' which encodes S3-relative paths
        used to derive input/target keys.
        """
        while True:
            selected_rows = self.df.sample(min(self.batch_size, len(self.df)))

            batch_inputs = []
            batch_labels = []

            for _, row in selected_rows.iterrows():
                if 'file_path' not in row:
                    # If your CSV schema differs, adapt here.
                    print("Row missing 'file_path'; skipping.")
                    continue

                # Build S3 keys from the dataframe path (aligning with train.py)
                try:
                    suffix = row.file_path.split('input', 1)[1]
                except Exception:
                    print(f"Unexpected file_path format: {row.file_path}; skipping.")
                    continue

                input_key = f"data/int5/input{suffix}"
                target_key = f"data/int5/mesh_swath_intervals{suffix}"

                inputs = self._load_npy_from_s3(input_key)
                targets = self._load_npy_from_s3(target_key)

                if inputs is None or targets is None:
                    print(f"Skipping invalid pair: {input_key} / {target_key}")
                    continue

                if inputs.ndim < 4:
                    inputs = np.expand_dims(inputs, axis=-1)

                # Reverse time to be consistent with train.py
                inputs = inputs[::-1, :, :, :]

                # Take first T steps and first input_channels channels
                inputs = inputs[:self.timesteps, :, :, :self.input_channels]

                # Normalize
                inputs = self.data_loader.normalize_field(inputs, self.norm_channels)
                inputs[np.isnan(inputs)] = 0.0

                # Targets: ensure 4D, slice to (T, H, W, 1) MESH channel
                if targets.ndim < 4:
                    targets = np.expand_dims(targets, axis=-1)
                targets = targets[:self.timesteps, :, :, 0:1]
                targets[np.isnan(targets)] = 0.0

                batch_inputs.append(inputs)
                batch_labels.append(targets)

            if batch_inputs:
                x = np.array(batch_inputs, dtype=np.float32)
                y = np.array(batch_labels, dtype=np.float32)
                x[np.isnan(x)] = 0.0
                y[np.isnan(y)] = 0.0
                yield x, y

    def _load_npy_from_s3(self, key):
        try:
            resp = self.data_loader.s3_client.get_object(Bucket=self.data_loader.bucket_name, Key=key)
            return np.load(BytesIO(resp['Body'].read()))
        except Exception as e:
            print(f"Error loading s3://{self.data_loader.bucket_name}/{key}: {e}")
            return None

    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(None, self.timesteps, self.height, self.width, self.input_channels),
                    dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(None, self.timesteps, self.height, self.width, self.output_channels),
                    dtype=tf.float32
                ),
            )
        ).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset


# ---------------- Model Trainer ----------------
class ModelTrainer:
    def __init__(self, args):
        self.args = args
        self.model = None

    def build_model(self):
        self.model = rnn(
            timesteps=self.args.timesteps,
            height=self.args.height,
            width=self.args.width,
            channels=self.args.input_channels,
            other_fields_shape=(self.args.timesteps, self.args.height,
                                self.args.width, self.args.input_channels - 1),
            initial_filters=self.args.initial_filters,
            final_activation='linear',
            dropout_rate=self.args.dropout_rate,
            l1_reg=self.args.l1_reg,
            l2_reg=self.args.l2_reg,
            x_pad=0,
            y_pad=0,
            kernel_size=self.args.kernel_size,
            padding='same',
            num_downsampling=self.args.num_downsampling,
            future_channels=self.args.input_channels - 1
        )
        print(f"Model built with {self.model.count_params():,} parameters")
        self.model.summary()
        return self.model

    def compile_model(self):
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.args.learning_rate,
            decay_steps=1000,
            alpha=0.1
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss=weighted_mse(), metrics=['accuracy', csi])

    def train(self, train_dataset, val_dataset, steps_per_epoch, val_steps):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self.args.version

        # SageMaker-friendly output dirs for logs/checkpoints
        checkpoint_dir = os.path.join(self.args.output_data_dir, 'checkpoints', version)
        log_dir = os.path.join(self.args.output_data_dir, 'logs', version)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
                monitor='val_csi', mode='max', save_best_only=True, verbose=1,
                initial_value_threshold=0.3
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_csi', mode='max', patience=40, verbose=1,
                restore_best_weights=True, start_from_epoch=40
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(log_dir, timestamp),
                histogram_freq=5, write_graph=True, update_freq='batch'
            ),
            keras.callbacks.CSVLogger(os.path.join(log_dir, f'{timestamp}.csv')),
        ]

        print(f"Starting training: {self.args.epochs} epochs, {steps_per_epoch} steps/epoch")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            verbose=1,
            callbacks=callbacks
        )

        # Save final model to SM_MODEL_DIR (SageMaker auto-uploads this)
        model_name = f'rnn_time-{timestamp}_version-{version}_filters{self.args.initial_filters}.keras'
        final_model_path = os.path.join(self.args.model_dir, model_name)
        os.makedirs(self.args.model_dir, exist_ok=True)
        self.model.save(final_model_path)
        print(f"Model saved to: {final_model_path}")

        # Also mirror under /opt/ml/output for convenience
        local_models_dir = os.path.join(self.args.output_data_dir, 'models', version)
        os.makedirs(local_models_dir, exist_ok=True)
        mirrored_path = os.path.join(local_models_dir, model_name)
        self.model.save(mirrored_path)
        print(f"Model mirrored to: {mirrored_path}")

        # Optional: upload to your bucket (matches train.py convention)
        try:
            s3 = boto3.client('s3')
            s3.upload_file(mirrored_path, self.args.bucket, f'models/{version}/{model_name}')
            print(f"Model uploaded to S3: s3://{self.args.bucket}/models/{version}/{model_name}")
        except Exception as e:
            print(f"WARNING: Failed to upload model to S3: {e}")

        # Persist history
        history_path = os.path.join(self.args.model_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

        return history


# ---------------- Main ----------------
def main():
    args = parse_args()

    print("="*80)
    print("SAGEMAKER CONVGRU TRAINING FOR MESH PREDICTION (train.py-aligned)")
    print("="*80)

    # Note on model_dir: SageMaker passed --model_dir (possibly S3). We save to SM_MODEL_DIR locally.
    if (args.model_dir or '').startswith('s3://'):
        print(f"[Info] model_dir provided as S3 path ('{args.model_dir}'). "
              f"Inside the container we will save to SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR', '/opt/ml/model')}")
        args.model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # GPU config
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Configured GPU: {gpu}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found, using CPU")

    # Print configuration
    print("\nTraining Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Initialize data utilities
    s3_client = boto3.client('s3')
    data_loader = DataLoader(
        args.bucket, s3_client,
        local_train_dir=args.train,
        local_val_dir=args.validation,
        norm_min_key=args.norm_min_key,
        norm_max_key=args.norm_max_key
    )

    # Load normalization params
    print("\nLoading normalization parameters...")
    data_loader.load_normalization_params()

    # Channels to normalize (all except binary mask at index 7) — mirror train.py
    norm_channels = [0, 1, 2, 3, 4, 5, 6]

    # Load dataframes from local channels
    print("Loading dataframes from local SageMaker channels...")
    train_df = data_loader.load_train_df(args.train_df_name)
    val_df = data_loader.load_val_df(args.val_df_name)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create datasets
    print("Creating datasets...")
    train_pipeline = InputPipeline(
        args.batch_size, data_loader, train_df, args.timesteps,
        args.height, args.width, args.input_channels, args.output_channels, norm_channels
    )
    val_pipeline = InputPipeline(
        args.batch_size, data_loader, val_df, args.timesteps,
        args.height, args.width, args.input_channels, args.output_channels, norm_channels
    )

    train_dataset = train_pipeline.create_dataset()
    val_dataset = val_pipeline.create_dataset()

    # Steps
    steps_per_epoch = max(1, len(train_df) // args.batch_size)
    val_steps = max(1, len(val_df) // args.batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {val_steps}")

    # Build / compile / train
    print("\nBuilding model...")
    trainer = ModelTrainer(args)
    trainer.build_model()
    trainer.compile_model()

    print("\nStarting training...")
    history = trainer.train(
        train_dataset,
        val_dataset,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps
    )

    # Final report
    final_val_csi = history.history.get('val_csi', [None])[-1]
    if final_val_csi is not None:
        print(f"\nFinal validation CSI: {final_val_csi:.4f}")
    else:
        print("\nFinal validation CSI not available in history.")
    print("Training complete!")

    gc.collect()


if __name__ == '__main__':
    main()
