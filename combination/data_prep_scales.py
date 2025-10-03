import numpy as np

import logging
logger = logging.getLogger("combiner.scale")

# Load the actual statistics from training
GLOBAL_MINS = np.load('global_mins.npy')
GLOBAL_MAXS = np.load('global_maxs.npy')

# Channel order from training
CHANNEL_ORDER = {
    'MESH_Max_60min': 0,
    'MESH': [1, 7],  # Appears at both indices
    'HeightCompositeReflectivity': 2,
    'EchoTop_50': 3,
    'PrecipRate': 4,
    'Reflectivity_0C': 5,
    'Reflectivity_-20C': 6
}

def scale_data_mrms(data, key, channel_idx=None):
    """
    Scale MRMS data to match training normalization.
      - Channel 1 (raw instantaneous MESH) stays UNscaled (clamped >= 0).
      - Channel 7 (dilated MESH) IS scaled like the others.
      - All outputs are clamped to a minimum of 0.0 (no 1e-5 floor).
    Logs pre/post stats when SCALE_LOG_LEVEL=DEBUG.
    """
    a = np.asarray(data, dtype=np.float32)
    # normalize key name like your code already does
    clean_key = key.replace('_00.50', '').replace('_00.00', '')

    # clip data att 0 to max
    a = np.clip(a, 0.0, a.max())
    logger.setLevel(logging.DEBUG)
    # Pre-stats for debug
    if logger.isEnabledFor(logging.DEBUG):
        pre_min = float(np.nanmin(a)) if a.size else float("nan")
        pre_max = float(np.nanmax(a)) if a.size else float("nan")
        pre_nan = int(np.isnan(a).sum())
        pre_neg = int((a < 0).sum())

    # Always enforce min 0.0 first (so raw MESH cannot go negative)
    a = np.clip(a, 0.0, None)

    # Resolve channel index if not provided
    if channel_idx is None:
        idx = CHANNEL_ORDER.get(clean_key, None)
        if isinstance(idx, (list, tuple)):
            # Avoid ambiguous [1,7] → caller must pass explicit index
            raise ValueError(
                f"Ambiguous field '{clean_key}' maps to {idx}; "
                f"pass channel_idx explicitly (1=raw MESH, 7=dilated MESH)."
            )
        channel_idx = idx

    # If still unknown → just return clamped data with a warning
    if channel_idx is None:
        logger.warning(f"Unknown field '{key}' (clean='{clean_key}'): returning unclipped/scaled={False}")
        return a

    # Channel 1 stays raw mm (already clamped)
    if channel_idx == 1:
        return a

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"[raw] {clean_key}[ch{channel_idx}] pre[min={pre_min:.3f},max={pre_max:.3f},neg={pre_neg},nan={pre_nan}] "
            f"→ post[min={float(np.nanmin(a)):.3f},max={float(np.nanmax(a)):.3f}]"
        )

    # Everything else → normalize with training mins/maxs, then clamp ≥0
    min_val = float(GLOBAL_MINS[channel_idx])
    max_val = float(GLOBAL_MAXS[channel_idx])
    
    denom = (max_val - min_val)
    if denom <= 0:
        logger.warning(f"Degenerate stats for {clean_key}[ch{channel_idx}] (mn={min_val}, mx={max_val}); returning zeros.")
        x = np.zeros_like(a, dtype=np.float32)
    else:
        x = (a - min_val) / (denom + 1e-6)
        x = np.clip(x, 0.0, None).astype(np.float32)

    if logger.isEnabledFor(logging.DEBUG):
        post_min = float(np.nanmin(x)) if x.size else float("nan")
        post_max = float(np.nanmax(x)) if x.size else float("nan")
        over1_pct = float(100.0 * np.mean(x > 1.0)) if x.size else 0.0
        logger.debug(
            f"[scale] {clean_key}[ch{channel_idx}] pre[min={pre_min:.3f},max={pre_max:.3f},neg={pre_neg},nan={pre_nan}] "
            f"mn={min_val:.3f} mx={max_val:.3f} → post[min={post_min:.3f},max={post_max:.3f},>1%={over1_pct:.4f}]"
        )
    return x

def scale_data_goes(data, key):
    """GOES scaling - you'll need to determine if these are part of your 8 channels"""
    # If GOES isn't in your 8 training channels, remove this
    # Otherwise map to appropriate channel indices
    return data

def scale_data_hrrr(data, key):
    """HRRR scaling - you'll need to determine if these are part of your 8 channels"""
    # If HRRR isn't in your 8 training channels, remove this
    # Otherwise map to appropriate channel indices
    return data