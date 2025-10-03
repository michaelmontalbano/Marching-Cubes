#!/usr/bin/env python3
"""
Evaluate MESH - 60-minute swath predictions

Evaluates model predictions against the 60-minute MESH swath target.
- Loads best_model_intervals.keras
- Evaluates all 12 prediction timesteps against the 60-minute swath
- Calculates CSI, POD, FAR, and Bias metrics
- Finds optimal prediction thresholds for each observation threshold
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------- TensorFlow / Keras & custom layers ----------------
import tensorflow as tf
from tensorflow import keras as tfk

# Import custom objects from modules - REQUIRED for model loading
try:
    from rnn import (
        reshape_and_stack, slice_to_n_steps, slice_output_shape,
        ResBlock, WarmUpCosineDecayScheduler, ConvGRU, ConvBlock,
        ZeroLikeLayer, ReflectionPadding2D, ResGRU, GRUResBlock
    )
    from models import weighted_mse, csi, focal_mse, combined_loss
    logger.info("Successfully imported custom objects from rnn and models modules")
except ImportError as e:
    logger.error(f"CRITICAL: Cannot import required modules - {e}")
    logger.error("Make sure rnn.py and models.py are in the same directory or in PYTHONPATH")
    raise ImportError("Required modules rnn.py and models.py not found. These are required for model loading.")

# ---------------- Channel indices ----------------
C_MESH60    = 0  # MESH_Max_60min (target channel)
C_MESH      = 1  # MESH (raw)
C_HCR       = 2  # HeightCompositeReflectivity
C_ECHOTOP50 = 3  # EchoTop_50
C_PRECIP    = 4  # PrecipRate
C_REF0C     = 5  # Reflectivity_0C
C_REFm20    = 6  # Reflectivity_-20C
C_MESH_DIL  = 7  # MESH dil mask (binary)
NORM_CHANNELS = (C_MESH60, C_MESH, C_HCR, C_ECHOTOP50, C_PRECIP, C_REF0C, C_REFm20)

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("evaluate_mesh_60min")

# ---------------- Config ----------------
class Config:
    # Buckets / paths
    MODEL_BUCKET       = os.getenv('MODEL_BUCKET', 'dev-grib-bucket')
    TEST_DF_BUCKET     = os.getenv('TEST_DF_BUCKET', MODEL_BUCKET)
    MODEL_S3_PATH      = os.getenv('MODEL_S3_PATH', 'models/best_model_intervals.keras')
    TEST_DF_PATH       = os.getenv('TEST_DF_PATH', 'dataframes/test.csv')
    NORMALIZATION_MIN_PATH = os.getenv('NORMALIZATION_MIN_PATH', 'global_mins3.npy')
    NORMALIZATION_MAX_PATH = os.getenv('NORMALIZATION_MAX_PATH', 'global_maxs3.npy')

    # Writable roots
    RESULTS_ROOT      = os.getenv('RESULTS_ROOT', './evaluation_results')
    MODEL_CACHE_ROOT  = os.getenv('MODEL_CACHE_ROOT', './model_cache')
    DATA_CACHE_ROOT   = os.getenv('DATA_CACHE_ROOT', './data')

    # Eval set
    NUM_SAMPLES = int(os.getenv('NUM_SAMPLES', '100'))

    # All timesteps (0-11) corresponding to 0-55 minutes in 5-minute intervals
    TIMESTEPS = np.arange(0, 12)
    CHANNELS  = np.arange(0, 8)

    # Thresholds (mm)
    OBS_THRESHOLDS = [5, 10, 20, 30, 40, 50, 70]
    
    # Prediction threshold search range
    PRED_CUTOFF_MIN  = 1.0
    PRED_CUTOFF_MAX  = 60.0
    PRED_CUTOFF_STEP = 1.0

    @property
    def MODEL_NAME(self):
        return os.path.splitext(os.path.basename(self.MODEL_S3_PATH))[0]
    @property
    def MODEL_CACHE_DIR(self):
        return os.path.join(self.MODEL_CACHE_ROOT, self.MODEL_NAME)
    @property
    def RESULTS_DIR(self):
        return os.path.join(self.RESULTS_ROOT, self.MODEL_NAME)

# ---------------- S3 Loader ----------------
class DataLoader:
    def __init__(self, s3, data_root: str):
        self.s3 = s3
        self.cache_dir = data_root
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, bucket: str, key: str) -> str:
        path = os.path.join(self.cache_dir, bucket, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def load_numpy(self, bucket: str, key: str) -> Optional[np.ndarray]:
        cp = self._cache_path(bucket, key)
        if os.path.exists(cp):
            try:
                return np.load(cp, allow_pickle=False)
            except Exception:
                try: os.remove(cp)
                except Exception: pass
        try:
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            data = obj['Body'].read()
            with open(cp, 'wb') as f: f.write(data)
            return np.load(BytesIO(data), allow_pickle=False)
        except Exception as e:
            logger.error(f"Failed to load numpy s3://{bucket}/{key}: {e}")
            return None

    def load_csv(self, bucket: str, key: str) -> Optional[pd.DataFrame]:
        cp = self._cache_path(bucket, key.replace('/', '_'))
        if os.path.exists(cp):
            try:
                return pd.read_csv(cp)
            except Exception:
                try: os.remove(cp)
                except Exception: pass
        try:
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            data = obj['Body'].read()
            df = pd.read_csv(BytesIO(data))
            df.to_csv(cp, index=False)
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV s3://{bucket}/{key}: {e}")
            return None

    @staticmethod
    def normalize_inputs(x: np.ndarray, gmins: np.ndarray, gmaxs: np.ndarray) -> np.ndarray:
        """Normalize channels in NORM_CHANNELS to ~[0,1]; mask stays 0/1."""
        x = x.astype(np.float32, copy=False)
        T, H, W, C = x.shape
        out = np.zeros_like(x, dtype=np.float32)
        for t in range(T):
            for c in range(C):
                arr = x[t, :, :, c]
                if c in NORM_CHANNELS:
                    mn, mx = float(gmins[c]), float(gmaxs[c])
                    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                        mn = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
                        mx = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else (mn + 1.0)
                        if mx <= mn: mx = mn + 1.0
                    out[t, :, :, c] = np.clip((arr - mn) / (mx - mn + 1e-5), 1e-5, 1.0)
                else:
                    out[t, :, :, c] = arr
        return out

# ---------------- Metrics Calculation ----------------
class MetricsCalculator:
    """Calculate metrics for 60-minute swath predictions"""
    
    def __init__(self, pred_thresholds, obs_thresholds):
        self.pred_thr = np.asarray(pred_thresholds, dtype=float)
        self.obs_thr = np.asarray(obs_thresholds, dtype=float)
        P, O = len(self.pred_thr), len(self.obs_thr)
        
        # Accumulate stats for each timestep separately
        self.timestep_stats = {}
        for t in range(12):
            self.timestep_stats[t] = {
                'tp': np.zeros((P, O), dtype=np.int64),
                'fp': np.zeros((P, O), dtype=np.int64),
                'fn': np.zeros((P, O), dtype=np.int64)
            }
        
        # Also keep overall stats
        self.overall_tp = np.zeros((P, O), dtype=np.int64)
        self.overall_fp = np.zeros((P, O), dtype=np.int64)
        self.overall_fn = np.zeros((P, O), dtype=np.int64)

    def update(self, predictions: np.ndarray, target_60min: np.ndarray):
        """
        Update metrics with a single sample's predictions and target.
        
        Args:
            predictions: (12, H, W, 1) - model predictions for each timestep
            target_60min: (H, W) - 60-minute MESH swath target
        """
        target = target_60min.ravel()
        
        # Process each timestep
        for t in range(predictions.shape[0]):
            pred_t = predictions[t].squeeze().ravel()
            
            for p_i, p_thr in enumerate(self.pred_thr):
                pred_pos = pred_t >= p_thr
                not_pred = ~pred_pos
                
                for o_i, o_thr in enumerate(self.obs_thr):
                    obs_pos = target >= o_thr
                    
                    tp = int(np.count_nonzero(pred_pos & obs_pos))
                    fp = int(np.count_nonzero(pred_pos & ~obs_pos))
                    fn = int(np.count_nonzero(not_pred & obs_pos))
                    
                    self.timestep_stats[t]['tp'][p_i, o_i] += tp
                    self.timestep_stats[t]['fp'][p_i, o_i] += fp
                    self.timestep_stats[t]['fn'][p_i, o_i] += fn
                    
                    # Add to overall stats
                    self.overall_tp[p_i, o_i] += tp
                    self.overall_fp[p_i, o_i] += fp
                    self.overall_fn[p_i, o_i] += fn

    def compute_metrics(self, tp, fp, fn):
        """Compute CSI, POD, FAR, and Bias from counts"""
        denom = tp + fp + fn
        with np.errstate(divide='ignore', invalid='ignore'):
            csi = np.where(denom > 0, tp / denom, 0.0)
            pod = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            far = np.where((tp + fp) > 0, fp / (tp + fp), 0.0)
            bias = np.where((tp + fn) > 0, (tp + fp) / (tp + fn), 0.0)
        return csi, pod, far, bias

    def finalize(self):
        """Compute final metrics for all timesteps and overall"""
        results = {}
        
        # Compute metrics for each timestep
        for t in range(12):
            stats = self.timestep_stats[t]
            csi, pod, far, bias = self.compute_metrics(stats['tp'], stats['fp'], stats['fn'])
            
            # Find best prediction threshold for each observation threshold
            best_idx = np.argmax(csi, axis=0)
            best_metrics = pd.DataFrame({
                "obs_threshold": self.obs_thr,
                "best_pred_threshold": self.pred_thr[best_idx],
                "CSI": csi[best_idx, np.arange(len(self.obs_thr))],
                "POD": pod[best_idx, np.arange(len(self.obs_thr))],
                "FAR": far[best_idx, np.arange(len(self.obs_thr))],
                "Bias": bias[best_idx, np.arange(len(self.obs_thr))]
            })
            
            results[f'timestep_{t}'] = {
                'minutes': t * 5,
                'best_metrics': best_metrics.to_dict(orient='records'),
                'csi_surface': csi.tolist(),
                'pod_surface': pod.tolist(),
                'far_surface': far.tolist(),
                'bias_surface': bias.tolist()
            }
        
        # Compute overall metrics (averaged across all timesteps)
        overall_csi, overall_pod, overall_far, overall_bias = self.compute_metrics(
            self.overall_tp / 12, self.overall_fp / 12, self.overall_fn / 12
        )
        
        best_idx_overall = np.argmax(overall_csi, axis=0)
        best_overall = pd.DataFrame({
            "obs_threshold": self.obs_thr,
            "best_pred_threshold": self.pred_thr[best_idx_overall],
            "CSI": overall_csi[best_idx_overall, np.arange(len(self.obs_thr))],
            "POD": overall_pod[best_idx_overall, np.arange(len(self.obs_thr))],
            "FAR": overall_far[best_idx_overall, np.arange(len(self.obs_thr))],
            "Bias": overall_bias[best_idx_overall, np.arange(len(self.obs_thr))]
        })
        
        results['overall'] = {
            'best_metrics': best_overall.to_dict(orient='records'),
            'csi_surface': overall_csi.tolist(),
            'pod_surface': overall_pod.tolist(),
            'far_surface': overall_far.tolist(),
            'bias_surface': overall_bias.tolist()
        }
        
        return results

# ---------------- Evaluator ----------------
class ModelEvaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.s3 = boto3.client('s3')
        self.loader = DataLoader(self.s3, cfg.DATA_CACHE_ROOT)
        self.model = None
        self.gmins = None
        self.gmaxs = None

        for d in (self.cfg.MODEL_CACHE_DIR, self.cfg.RESULTS_DIR):
            os.makedirs(d, exist_ok=True)

    def load_model(self) -> bool:
        cache_path = os.path.join(self.cfg.MODEL_CACHE_DIR, f"{self.cfg.MODEL_NAME}.keras")
        logger.info(f"Model cache path: {cache_path}")
        try:
            if not os.path.exists(cache_path):
                logger.info(f"Downloading model: s3://{self.cfg.MODEL_BUCKET}/{self.cfg.MODEL_S3_PATH}")
                self.s3.download_file(self.cfg.MODEL_BUCKET, self.cfg.MODEL_S3_PATH, cache_path)

            # Custom objects for model loading - MUST use actual classes from rnn.py and models.py
            custom_objects = {
                # Loss functions
                'loss': weighted_mse(),
                'weighted_mse': weighted_mse(),
                'focal_mse': focal_mse(),
                'combined_loss': combined_loss(),
                'csi': csi,
                # Helper functions
                'reshape_and_stack': reshape_and_stack,
                'slice_to_n_steps': slice_to_n_steps,
                'slice_output_shape': slice_output_shape,
                # Custom layers from rnn.py
                'ResBlock': ResBlock,
                'ConvBlock': ConvBlock,
                'ZeroLikeLayer': ZeroLikeLayer,
                'ReflectionPadding2D': ReflectionPadding2D,
                'ConvGRU': ConvGRU,
                'ResGRU': ResGRU,
                'GRUResBlock': GRUResBlock,
                # Learning rate scheduler
                'WarmUpCosineDecayScheduler': WarmUpCosineDecayScheduler,
            }
            
            # Load model with custom objects
            self.model = tf.keras.models.load_model(cache_path, custom_objects=custom_objects, compile=False)
            
            # Log model info
            logger.info(f"Loaded model: {self.cfg.MODEL_NAME}")
            logger.info(f"  Input shape:  {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")
            logger.info(f"  Parameters:   {self.model.count_params():,}")
            
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            logger.error(f"Make sure rnn.py and models.py are in the same directory as this script")
            return False

    def load_norm(self) -> bool:
        self.gmins = self.loader.load_numpy(self.cfg.MODEL_BUCKET, self.cfg.NORMALIZATION_MIN_PATH)
        self.gmaxs = self.loader.load_numpy(self.cfg.MODEL_BUCKET, self.cfg.NORMALIZATION_MAX_PATH)
        ok = self.gmins is not None and self.gmaxs is not None
        if not ok:
            logger.error("Normalization params not found.")
        else:
            logger.info(f"Loaded normalization params: mins shape={self.gmins.shape}, maxs shape={self.gmaxs.shape}")
        return ok

    @staticmethod
    def _resolve_paths_from_row(row: pd.Series) -> Optional[Tuple[str, str]]:
        """Resolve input and target paths from dataframe row"""
        target_key = None
        for cand in ('target_path', 'target', 'y_path'):
            if cand in row and isinstance(row[cand], str) and row[cand].strip():
                target_key = row[cand].strip()
                break

        input_key = None
        for cand in ('file_path', 'input_path', 'inputs_path', 'x_path'):
            if cand in row and isinstance(row[cand], str) and row[cand].strip():
                input_key = row[cand].strip()
                break

        if input_key is None and target_key is None:
            return None

        if target_key is None and input_key is not None:
            # Try to infer target path from input path
            if "/input/" in input_key:
                target_key = input_key.replace("/input/", "/mesh_swath_intervals/")
            elif "input/" in input_key:
                target_key = input_key.replace("input/", "mesh_swath_intervals/")
            else:
                if "input" in input_key:
                    suffix = input_key.split("input", 1)[1]
                    target_key = f"data/int5/mesh_swath_intervals{suffix}"
                else:
                    base = os.path.basename(input_key)
                    target_key = input_key.replace(base, f"mesh_swath_intervals_{base}")
        
        return input_key, target_key

    def extract_60min_swath(self, target_array: np.ndarray) -> np.ndarray:
        """
        Extract 60-minute MESH swath from target array.
        The 60-minute swath is typically in channel 0 of the target.
        """
        if target_array.ndim == 4:
            # Shape: (T, H, W, C) - take channel 0 at any timestep (they should be the same)
            return target_array[0, :, :, 0]
        elif target_array.ndim == 3:
            # Shape: (T, H, W) or (H, W, C)
            if target_array.shape[0] == 12:  # Likely (T, H, W)
                return target_array[0, :, :]
            else:  # Likely (H, W, C)
                return target_array[:, :, 0]
        elif target_array.ndim == 2:
            # Already 2D
            return target_array
        else:
            logger.warning(f"Unexpected target shape: {target_array.shape}")
            return target_array.squeeze()[:, :, 0] if target_array.shape[-1] > 1 else target_array.squeeze()

    def evaluate(self) -> Dict:
        df = self.loader.load_csv(self.cfg.TEST_DF_BUCKET, self.cfg.TEST_DF_PATH)
        if df is None or df.empty:
            logger.error("Empty test dataframe.")
            return {}
        
        # Sample if needed
        if len(df) > self.cfg.NUM_SAMPLES:
            df = df.sample(self.cfg.NUM_SAMPLES, random_state=42)
            logger.info(f"Sampled {self.cfg.NUM_SAMPLES} from {len(df)} total samples")
        
        logger.info(f"Evaluating {len(df)} samples against 60-minute MESH swath")

        # Setup metrics calculator
        pred_cuts = np.arange(self.cfg.PRED_CUTOFF_MIN, self.cfg.PRED_CUTOFF_MAX + 1e-9, self.cfg.PRED_CUTOFF_STEP)
        obs_thrs = np.array(self.cfg.OBS_THRESHOLDS, dtype=float)
        
        metrics_calc = MetricsCalculator(pred_cuts, obs_thrs)
        
        # Track statistics
        successful_samples = 0
        failed_samples = 0
        prediction_stats = []
        target_stats = []
        
        # Main evaluation loop
        for i, (idx, row) in enumerate(df.iterrows(), start=1):
            if i % 20 == 0:
                logger.info(f"Progress: {i}/{len(df)} samples")

            resolved = self._resolve_paths_from_row(row)
            if not resolved:
                logger.warning(f"Row {idx}: could not resolve paths")
                failed_samples += 1
                continue
            input_key, target_key = resolved

            # Load data
            x = self.loader.load_numpy(self.cfg.MODEL_BUCKET, input_key)
            y = self.loader.load_numpy(self.cfg.MODEL_BUCKET, target_key)
            if x is None or y is None:
                logger.warning(f"Row {idx}: missing arrays")
                failed_samples += 1
                continue

            # Extract 60-minute swath from target
            target_60min = self.extract_60min_swath(y)
            
            # Process input
            if x.ndim < 4:
                x = np.expand_dims(x, -1)
            x = x[::-1, :, :, :]  # reverse time
            x = x[self.cfg.TIMESTEPS][:, :, :, self.cfg.CHANNELS]
            
            # Normalize inputs
            x = self.loader.normalize_inputs(x, self.gmins, self.gmaxs)
            x[np.isnan(x)] = 0.0
            target_60min[np.isnan(target_60min)] = 0.0

            # Predict
            try:
                pred = self.model.predict(np.expand_dims(x, 0), verbose=0)  # (1, 12, H, W, 1)
                pred = pred[0]  # (12, H, W, 1)
                successful_samples += 1
            except Exception as e:
                logger.error(f"Prediction failed for sample {idx}: {e}")
                failed_samples += 1
                continue

            # Update metrics
            metrics_calc.update(pred, target_60min)
            
            # Collect statistics
            prediction_stats.append({
                'min': float(np.min(pred)),
                'max': float(np.max(pred)),
                'mean': float(np.mean(pred)),
                'std': float(np.std(pred))
            })
            target_stats.append({
                'min': float(np.min(target_60min)),
                'max': float(np.max(target_60min)),
                'mean': float(np.mean(target_60min)),
                'std': float(np.std(target_60min))
            })

        logger.info(f"Evaluation complete: {successful_samples} successful, {failed_samples} failed")

        # Finalize metrics
        metrics_results = metrics_calc.finalize()
        
        # Compute aggregate statistics
        if prediction_stats:
            agg_pred_stats = {
                'min': float(np.mean([s['min'] for s in prediction_stats])),
                'max': float(np.mean([s['max'] for s in prediction_stats])),
                'mean': float(np.mean([s['mean'] for s in prediction_stats])),
                'std': float(np.mean([s['std'] for s in prediction_stats]))
            }
        else:
            agg_pred_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        if target_stats:
            agg_target_stats = {
                'min': float(np.mean([s['min'] for s in target_stats])),
                'max': float(np.mean([s['max'] for s in target_stats])),
                'mean': float(np.mean([s['mean'] for s in target_stats])),
                'std': float(np.mean([s['std'] for s in target_stats]))
            }
        else:
            agg_target_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        # Create final results
        final_results = {
            'model': self.cfg.MODEL_NAME,
            'evaluation_datetime': datetime.utcnow().isoformat(),
            'num_samples': len(df),
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'target': '60-minute MESH swath',
            'prediction_statistics': agg_pred_stats,
            'target_statistics': agg_target_stats,
            'metrics_by_timestep': metrics_results,
            'observation_thresholds': self.cfg.OBS_THRESHOLDS,
            'prediction_threshold_range': {
                'min': self.cfg.PRED_CUTOFF_MIN,
                'max': self.cfg.PRED_CUTOFF_MAX,
                'step': self.cfg.PRED_CUTOFF_STEP
            }
        }
        
        # Print summary
        self._print_summary(metrics_results)
        
        return final_results

    def _print_summary(self, metrics_results):
        """Print evaluation summary"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY - 60-MINUTE MESH SWATH")
        logger.info("="*80)
        
        # Print overall metrics
        overall = metrics_results['overall']['best_metrics']
        logger.info("\nOVERALL METRICS (averaged across all timesteps):")
        logger.info(f"{'Obs Thr':>8} {'Best Pred':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8}")
        logger.info("-" * 60)
        for metric in overall:
            logger.info(f"{metric['obs_threshold']:>7.0f}mm {metric['best_pred_threshold']:>9.1f}mm "
                       f"{metric['CSI']:>8.3f} {metric['POD']:>8.3f} {metric['FAR']:>8.3f} {metric['Bias']:>8.2f}")
        
        # Print metrics for key timesteps
        key_timesteps = [0, 6, 11]  # 0, 30, and 55 minutes
        for t in key_timesteps:
            timestep_key = f'timestep_{t}'
            if timestep_key in metrics_results:
                minutes = metrics_results[timestep_key]['minutes']
                metrics = metrics_results[timestep_key]['best_metrics']
                
                logger.info(f"\nTIMESTEP {t} ({minutes} minutes) METRICS:")
                logger.info(f"{'Obs Thr':>8} {'Best Pred':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8}")
                logger.info("-" * 60)
                for metric in metrics:
                    logger.info(f"{metric['obs_threshold']:>7.0f}mm {metric['best_pred_threshold']:>9.1f}mm "
                               f"{metric['CSI']:>8.3f} {metric['POD']:>8.3f} {metric['FAR']:>8.3f} {metric['Bias']:>8.2f}")

# ---------------- Main ----------------
def main():
    cfg = Config()
    logger.info("="*80)
    logger.info("EVALUATE MESH MODEL - 60-MINUTE SWATH TARGET")
    logger.info("="*80)
    logger.info(f"Model:     {cfg.MODEL_NAME}")
    logger.info(f"Model S3:  s3://{cfg.MODEL_BUCKET}/{cfg.MODEL_S3_PATH}")
    logger.info(f"Test CSV:  s3://{cfg.TEST_DF_BUCKET}/{cfg.TEST_DF_PATH}")
    logger.info(f"Samples:   {cfg.NUM_SAMPLES}")
    logger.info(f"Target:    60-minute MESH swath")
    logger.info(f"Results dir: {cfg.RESULTS_DIR}")
    logger.info("="*80)

    # Create directories
    for d in (cfg.MODEL_CACHE_DIR, cfg.RESULTS_DIR, cfg.DATA_CACHE_ROOT):
        os.makedirs(d, exist_ok=True)

    # Initialize evaluator
    ev = ModelEvaluator(cfg)
    
    # Load model and normalization
    if not ev.load_model(): 
        logger.error("Failed to load model")
        return
    if not ev.load_norm():  
        logger.error("Failed to load normalization parameters")
        return

    # Run evaluation
    results = ev.evaluate()
    if not results:
        logger.error("No results produced.")
        return

    # Save results
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(cfg.RESULTS_DIR, f"results_60min_swath_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {out_json}")

    # Save summary CSV for easy viewing
    if 'overall' in results.get('metrics_by_timestep', {}):
        overall_df = pd.DataFrame(results['metrics_by_timestep']['overall']['best_metrics'])
        csv_path = os.path.join(cfg.RESULTS_DIR, f"overall_metrics_{ts}.csv")
        overall_df.to_csv(csv_path, index=False)
        logger.info(f"Overall metrics CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()