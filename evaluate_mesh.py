#!/usr/bin/env python3
"""
Evaluate MESH - 60-minute swath predictions with multi-channel support

Evaluates model predictions against the 60-minute MESH swath target.
Supports models with 6 or 21 input channels.
- For 6-channel models: uses first 6 channels from data
- For 21-channel models: uses first 6 channels + expands to 21
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
logger = logging.getLogger("evaluate_mesh_60min")

# Import custom objects from modules
try:
    from rnn import (
        reshape_and_stack, slice_to_n_steps, slice_output_shape,
        ResBlock, WarmUpCosineDecayScheduler, ConvGRU, ConvBlock,
        ZeroLikeLayer, ReflectionPadding2D, ResGRU, GRUResBlock
    )
    from models import weighted_mse, csi 
    logger.info("Successfully imported custom objects from rnn and models modules")
except ImportError as e:
    logger.error(f"CRITICAL: Cannot import required modules - {e}")
    raise ImportError("Required modules rnn.py and models.py not found.")

# ---------------- Channel indices ----------------
C_MESH60    = 0  # MESH_Max_60min (target channel)
C_MESH      = 1  # MESH (raw)
C_HCR       = 2  # HeightCompositeReflectivity
C_ECHOTOP50 = 3  # EchoTop_50
C_PRECIP    = 4  # PrecipRate
C_REF0C     = 5  # Reflectivity_0C
C_REFm20    = 6  # Reflectivity_-20C
C_MESH_DIL  = 7  # MESH dil mask (binary)

# First 6 channels for 6-channel models
BASE_CHANNELS_6 = [C_MESH60, C_MESH, C_HCR, C_ECHOTOP50, C_PRECIP, C_REF0C]
NORM_CHANNELS_6 = [C_MESH60, C_MESH, C_HCR, C_ECHOTOP50, C_PRECIP, C_REF0C]

# 21-channel model uses first 6 + extended features
NORM_CHANNELS_21 = list(range(21))  # Normalize all 21 channels

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
    MODEL_DIR          = os.getenv('MODEL_DIR', './models')  # Local directory containing models
    MODEL_PATTERN      = os.getenv('MODEL_PATTERN', 'growth*/*/*.keras')  # Pattern within MODEL_DIR
    TEST_DF_PATH       = os.getenv('TEST_DF_PATH', 'dataframes/test.csv')
    NORMALIZATION_MIN_PATH = os.getenv('NORMALIZATION_MIN_PATH', 'global_mins.npy')
    NORMALIZATION_MAX_PATH = os.getenv('NORMALIZATION_MAX_PATH', 'global_maxs.npy')

    # Writable roots
    RESULTS_ROOT      = os.getenv('RESULTS_ROOT', './evaluation_results')
    MODEL_CACHE_ROOT  = os.getenv('MODEL_CACHE_ROOT', './model_cache')
    DATA_CACHE_ROOT   = os.getenv('DATA_CACHE_ROOT', './data')

    # Eval set
    NUM_SAMPLES = int(os.getenv('NUM_SAMPLES', '100'))

    # All timesteps (0-11) corresponding to 0-55 minutes in 5-minute intervals
    TIMESTEPS = np.arange(0, 12)
    
    # Thresholds (mm)
    OBS_THRESHOLDS = [5, 10, 20, 30, 40, 50, 70]
    
    # Prediction threshold search range
    PRED_CUTOFF_MIN  = 1.0
    PRED_CUTOFF_MAX  = 60.0
    PRED_CUTOFF_STEP = 1.0

    @property
    def MODEL_NAME(self):
        # Will be set dynamically per model
        if hasattr(self, '_current_model_path'):
            return os.path.splitext(os.path.basename(self._current_model_path))[0]
        return 'unknown'
    
    @property
    def RESULTS_DIR(self):
        return os.path.join(self.RESULTS_ROOT, self.MODEL_NAME)
    
    def set_current_model(self, model_path: str):
        """Set the current model path for property methods"""
        self._current_model_path = model_path

# ---------------- Local Model Discovery ----------------
def find_local_models(model_dir: str, pattern: str) -> List[str]:
    """
    Find all model files locally matching the pattern.
    
    Args:
        model_dir: Local directory containing models (e.g., './models')
        pattern: Pattern like 'growth*/*/*.keras'
    
    Returns:
        List of local file paths matching the pattern
    """
    import glob
    
    full_pattern = os.path.join(model_dir, pattern)
    logger.info(f"Searching for models: {full_pattern}")
    
    models = glob.glob(full_pattern)
    models = sorted(models)
    
    logger.info(f"Found {len(models)} models matching pattern")
    return models

def detect_model_channels_from_path(model_path: str) -> int:
    """
    Infer expected channels from model path.
    
    Rules:
    - growth_actual, growth_all, growth_half: 21 channels (full feature set)
    - growth, growth2: 6 channels (base features)
    
    Note: This is a heuristic and should be verified against actual model input shape.
    """
    path_lower = model_path.lower()
    
    if any(x in path_lower for x in ['growth_actual', 'growth_all', 'growth_half']):
        return 21
    elif any(x in path_lower for x in ['growth2', '/growth/']):
        return 6
    else:
        # Default to 6 for unknown patterns
        logger.warning(f"Cannot infer channels from path {model_path}, will use model's actual input shape")
        return None

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
    def normalize_inputs(x: np.ndarray, gmins: np.ndarray, gmaxs: np.ndarray, 
                        norm_channels: List[int]) -> np.ndarray:
        """Normalize specified channels to ~[0,1]; others stay unchanged."""
        x = x.astype(np.float32, copy=False)
        T, H, W, C = x.shape
        out = np.zeros_like(x, dtype=np.float32)
        
        for t in range(T):
            for c in range(C):
                arr = x[t, :, :, c]
                if c in norm_channels and c < len(gmins):
                    mn, mx = float(gmins[c]), float(gmaxs[c])
                    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                        mn = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
                        mx = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else (mn + 1.0)
                        if mx <= mn: mx = mn + 1.0
                    out[t, :, :, c] = np.clip((arr - mn) / (mx - mn + 1e-5), 1e-5, 1.0)
                else:
                    out[t, :, :, c] = arr
        return out

# ---------------- Channel Adapter ----------------
def adapt_channels_for_model(x: np.ndarray, target_channels: int) -> np.ndarray:
    """
    Adapt input data to match model's expected channel count.
    
    Args:
        x: Input array of shape (T, H, W, C)
        target_channels: Number of channels the model expects (6, 8, or 21)
    
    Returns:
        Array with shape (T, H, W, target_channels)
    """
    T, H, W, C = x.shape
    
    if C == target_channels:
        return x
    
    if target_channels == 6:
        # Use first 6 channels
        logger.info(f"Adapting {C} channels → 6 channels (using first 6)")
        return x[:, :, :, :6]
    
    elif target_channels == 8:
        # Use all 8 channels from data
        if C >= 8:
            logger.info(f"Adapting {C} channels → 8 channels (using first 8)")
            return x[:, :, :, :8]
        else:
            # Pad with zeros if we have fewer than 8
            logger.info(f"Adapting {C} channels → 8 channels (using {C} + {8-C} zero channels)")
            adapted = np.zeros((T, H, W, 8), dtype=x.dtype)
            adapted[:, :, :, :C] = x
            return adapted
    
    elif target_channels == 21:
        # Use first 6 channels and pad with zeros to reach 21
        logger.info(f"Adapting {C} channels → 21 channels (first 6 + 15 zero channels)")
        adapted = np.zeros((T, H, W, 21), dtype=x.dtype)
        # Copy first 6 channels
        adapted[:, :, :, :6] = x[:, :, :, :6]
        # Remaining 15 channels stay as zeros
        return adapted
    
    else:
        raise ValueError(f"Unsupported target_channels: {target_channels}. Only 6, 8, or 21 supported.")

def detect_model_channels(model) -> int:
    """
    Detect expected number of channels from model input shape.
    
    Args:
        model: Loaded Keras model
    
    Returns:
        Number of expected channels (6, 8, or 21)
    """
    input_shape = model.input_shape
    
    # Handle different input shape formats
    if isinstance(input_shape, list):
        # Multiple inputs - use first one
        channels = input_shape[0][-1]
    else:
        # Single input
        channels = input_shape[-1]
    
    logger.info(f"Detected model expects {channels} channels from input shape")
    
    return channels

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
        """Update metrics with a single sample's predictions and target."""
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
        self.model_channels = None
        self.current_model_path = None

        os.makedirs(cfg.DATA_CACHE_ROOT, exist_ok=True)

    def load_model(self, model_path: str) -> bool:
        """Load a specific model from local filesystem"""
        self.current_model_path = model_path
        self.cfg.set_current_model(model_path)
        
        # Create results directory
        os.makedirs(self.cfg.RESULTS_DIR, exist_ok=True)
        
        logger.info(f"Loading model from: {model_path}")
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            # Custom objects for model loading
            custom_objects = {
                'loss': weighted_mse(),
                'weighted_mse': weighted_mse(),
                'csi': csi,
                'reshape_and_stack': reshape_and_stack,
                'slice_to_n_steps': slice_to_n_steps,
                'slice_output_shape': slice_output_shape,
                'ResBlock': ResBlock,
                'ConvBlock': ConvBlock,
                'ZeroLikeLayer': ZeroLikeLayer,
                'ReflectionPadding2D': ReflectionPadding2D,
                'ConvGRU': ConvGRU,
                'ResGRU': ResGRU,
                'GRUResBlock': GRUResBlock,
                'WarmUpCosineDecayScheduler': WarmUpCosineDecayScheduler,
            }
            
            # Load model
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            
            # Detect expected channels from model's actual input shape
            self.model_channels = detect_model_channels(self.model)
            
            # Also check path-based inference for logging
            expected_from_path = detect_model_channels_from_path(model_path)
            if expected_from_path and self.model_channels != expected_from_path:
                logger.warning(f"Channel mismatch: path suggests {expected_from_path}, model actually has {self.model_channels}")
                logger.warning(f"Using model's actual channel count: {self.model_channels}")
            
            # Log model info
            logger.info(f"Loaded model: {self.cfg.MODEL_NAME}")
            logger.info(f"  Model path: {model_path}")
            logger.info(f"  Input shape:  {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")
            logger.info(f"  Expected channels: {self.model_channels}")
            logger.info(f"  Parameters:   {self.model.count_params():,}")
            
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
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
        """Extract 60-minute MESH swath from target array."""
        if target_array.ndim == 4:
            return target_array[0, :, :, 0]
        elif target_array.ndim == 3:
            if target_array.shape[0] == 12:
                return target_array[0, :, :]
            else:
                return target_array[:, :, 0]
        elif target_array.ndim == 2:
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
        logger.info(f"Model expects {self.model_channels} channels")

        # Setup metrics calculator
        pred_cuts = np.arange(self.cfg.PRED_CUTOFF_MIN, self.cfg.PRED_CUTOFF_MAX + 1e-9, self.cfg.PRED_CUTOFF_STEP)
        obs_thrs = np.array(self.cfg.OBS_THRESHOLDS, dtype=float)
        
        metrics_calc = MetricsCalculator(pred_cuts, obs_thrs)
        
        # Track statistics
        successful_samples = 0
        failed_samples = 0
        prediction_stats = []
        target_stats = []
        
        # Select normalization channels based on model
        if self.model_channels == 21:
            norm_channels = NORM_CHANNELS_21
        elif self.model_channels == 8:
            norm_channels = list(range(8))  # Normalize all 8 channels
        else:  # 6 channels
            norm_channels = NORM_CHANNELS_6
        
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
            x = x[self.cfg.TIMESTEPS]
            
            # Log original shape
            logger.debug(f"Original data shape: {x.shape}")
            
            # Adapt channels to match model requirements BEFORE normalization
            x = adapt_channels_for_model(x, self.model_channels)
            logger.debug(f"After adaptation shape: {x.shape}")
            
            # Normalize inputs
            x = self.loader.normalize_inputs(x, self.gmins, self.gmaxs, norm_channels)
            x[np.isnan(x)] = 0.0
            target_60min[np.isnan(target_60min)] = 0.0

            # Predict
            try:
                pred = self.model.predict(np.expand_dims(x, 0), verbose=0)
                pred = pred[0]
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
            'model_channels': self.model_channels,
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
        logger.info(f"EVALUATION SUMMARY - 60-MINUTE MESH SWATH ({self.model_channels} channels)")
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
        key_timesteps = [0, 6, 11]
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
    logger.info("EVALUATE MESH MODELS - 60-MINUTE SWATH TARGET (MULTI-CHANNEL)")
    logger.info("="*80)
    logger.info(f"Model Dir:     {cfg.MODEL_DIR}")
    logger.info(f"Model Pattern: {cfg.MODEL_PATTERN}")
    logger.info(f"Test CSV:      s3://{cfg.TEST_DF_BUCKET}/{cfg.TEST_DF_PATH}")
    logger.info(f"Samples:       {cfg.NUM_SAMPLES}")
    logger.info(f"Target:        60-minute MESH swath")
    logger.info("="*80)

    # Create base directories
    os.makedirs(cfg.DATA_CACHE_ROOT, exist_ok=True)
    os.makedirs(cfg.RESULTS_ROOT, exist_ok=True)

    # Initialize evaluator
    ev = ModelEvaluator(cfg)
    
    # Load normalization parameters once (shared across all models)
    if not ev.load_norm():  
        logger.error("Failed to load normalization parameters")
        return

    # Find all models matching the pattern
    model_paths = find_local_models(cfg.MODEL_DIR, cfg.MODEL_PATTERN)
    
    if not model_paths:
        logger.error(f"No models found matching pattern: {os.path.join(cfg.MODEL_DIR, cfg.MODEL_PATTERN)}")
        return
    
    logger.info(f"\nFound {len(model_paths)} models to evaluate:")
    for i, path in enumerate(model_paths, 1):
        logger.info(f"  {i}. {path}")
    logger.info("")

    # Track results across all models
    all_results = []
    successful_models = 0
    failed_models = 0
    
    # Evaluate each model
    for i, model_path in enumerate(model_paths, 1):
        logger.info("\n" + "="*80)
        logger.info(f"EVALUATING MODEL {i}/{len(model_paths)}")
        logger.info("="*80)
        
        try:
            # Load model
            if not ev.load_model(model_path):
                logger.error(f"Failed to load model: {model_path}")
                failed_models += 1
                continue
            
            # Run evaluation
            results = ev.evaluate()
            if not results:
                logger.error(f"No results produced for model: {model_path}")
                failed_models += 1
                continue
            
            # Save individual model results
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_json = os.path.join(cfg.RESULTS_DIR, f"results_60min_swath_{ts}.json")
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to: {out_json}")
            
            # Save summary CSV
            if 'overall' in results.get('metrics_by_timestep', {}):
                overall_df = pd.DataFrame(results['metrics_by_timestep']['overall']['best_metrics'])
                csv_path = os.path.join(cfg.RESULTS_DIR, f"overall_metrics_{ts}.csv")
                overall_df.to_csv(csv_path, index=False)
                logger.info(f"Overall metrics CSV saved to: {csv_path}")
            
            # Add to combined results
            all_results.append({
                'model_path': model_path,
                'model_name': cfg.MODEL_NAME,
                'model_channels': ev.model_channels,
                'results': results
            })
            successful_models += 1
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_path}: {e}", exc_info=True)
            failed_models += 1
            continue
    
    # Save combined results summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE - ALL MODELS")
    logger.info("="*80)
    logger.info(f"Total models:      {len(model_paths)}")
    logger.info(f"Successful:        {successful_models}")
    logger.info(f"Failed:            {failed_models}")
    
    if all_results:
        # Create combined summary
        combined_summary = {
            'evaluation_datetime': datetime.utcnow().isoformat(),
            'total_models': len(model_paths),
            'successful_models': successful_models,
            'failed_models': failed_models,
            'model_directory': cfg.MODEL_DIR,
            'model_pattern': cfg.MODEL_PATTERN,
            'num_samples_per_model': cfg.NUM_SAMPLES,
            'models': []
        }
        
        # Extract key metrics for each model
        for result in all_results:
            model_info = {
                'model_path': result['model_path'],
                'model_name': result['model_name'],
                'channels': result['model_channels'],
                'successful_samples': result['results'].get('successful_samples', 0),
                'failed_samples': result['results'].get('failed_samples', 0)
            }
            
            # Add overall best metrics
            if 'overall' in result['results'].get('metrics_by_timestep', {}):
                overall_metrics = result['results']['metrics_by_timestep']['overall']['best_metrics']
                # Extract CSI at key thresholds
                model_info['best_metrics'] = {
                    f"csi_at_{int(m['obs_threshold'])}mm": m['CSI'] 
                    for m in overall_metrics
                }
            
            combined_summary['models'].append(model_info)
        
        # Save combined summary
        combined_path = os.path.join(cfg.RESULTS_ROOT, f"combined_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(combined_path, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        logger.info(f"\nCombined summary saved to: {combined_path}")
        
        # Create comparison table
        comparison_rows = []
        for model_info in combined_summary['models']:
            row = {
                'Model': model_info['model_name'],
                'Channels': model_info['channels'],
                'Success': model_info['successful_samples'],
                'Failed': model_info['failed_samples']
            }
            if 'best_metrics' in model_info:
                for key, val in model_info['best_metrics'].items():
                    row[key.replace('csi_at_', 'CSI@')] = f"{val:.3f}"
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_csv = os.path.join(cfg.RESULTS_ROOT, f"model_comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        logger.info(f"Model comparison CSV saved to: {comparison_csv}")
        
        # Print comparison table
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        logger.info("\n" + comparison_df.to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SCRIPT COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()