#!/usr/bin/env python3
"""
Combines recent MRMS timesteps, writes ground-truth swaths, and prepares model inputs.

Key fixes:
- No early use of mesh_max_60min before assignment.
- Compute mesh_max_60min AFTER building mesh_cache, for EVERY target timestep.
- Deduplicate GT writes per cycle and also skip if object already exists in S3.
- Keep channel layout: index 7 is "Dilated MESH" (second 'MESH' entry).
"""

import os
import time
import pickle
from io import BytesIO
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError
import numpy as np
import scipy.ndimage
import lz4.frame
import redis
import logging
from dotenv import load_dotenv
load_dotenv()
from scipy.ndimage import binary_dilation

from data_prep_scales import scale_data_mrms
# Main logger for this process
logger = logging.getLogger("combiner")

# Handlers only once
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler (capture everything, incl. DEBUG)
    fh = logging.FileHandler("data_combiner.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler (level configurable; default INFO)
    console_level = getattr(logging, os.getenv("COMBINER_CONSOLE_LEVEL", "DEBUG").upper(), logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

# Logger level (gate for non-child loggers)
logger.setLevel(getattr(logging, os.getenv("COMBINER_LOG_LEVEL", "DEBUG").upper(), logging.INFO))
# Do not bubble to root (avoids duplicate lines when something else configures root)
logger.propagate = False

# Scaler child logger inherits the same handlers; only set its level
scale_level = getattr(logging, os.getenv("SCALE_LOG_LEVEL", "DEBUG").upper(), logging.INFO)
logging.getLogger("combiner.scale").setLevel(scale_level)
# ----------------- Env / AWS -----------------

mrms_bucket = os.getenv("MRMS_BUCKET")
output_bucket = os.getenv("RAW_PRODUCT_BUCKET")

s3_client = boto3.client("s3")
s3 = boto3.resource("s3")

# ----------------- Redis -----------------
try:
    redis_host = os.getenv("REDIS_URL_0", "localhost")
    r_cache = redis.Redis(host=redis_host, port=6379)
    r_cache.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis cache connected")
except Exception:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available")

# ----------------- Config -----------------
# Channel order preserves your prior convention: index 7 is the "Dilated MESH" duplicate.
mrms_keys = ['MESH_Max_60min','MESH','HeightCompositeReflectivity','EchoTop_50',
             'PrecipRate','Reflectivity_0C','Reflectivity_-20C','MESH']
mrms_keys = ["mrms_" + k for k in mrms_keys]

TARGET_H = 1750
TARGET_W = 3500

# ----------------- Helpers -----------------
def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def download_data(bucket_name: str, key: str):
    """Download one MRMS pickle (LZ4 compressed) from S3 and return dict of fields."""
    bucket = s3.Bucket(bucket_name)
    file_obj = bucket.Object(key).get()["Body"].read()
    with lz4.frame.open(BytesIO(file_obj)) as s3_pickle:
        return pickle.loads(s3_pickle.read())

def get_mrms_files():
    """Return latest 120 minutes of MRMS keys (most-recent first)."""
    current_time = datetime.utcnow()
    start_time = current_time - timedelta(minutes=120)

    # Cache the listing briefly to reduce S3 calls.
    if REDIS_AVAILABLE:
        cached = r_cache.get("mrms_file_list")
        if cached:
            ts = r_cache.get("mrms_file_list_timestamp")
            if ts and time.time() - float(ts) < 240:
                return pickle.loads(cached)

    files = []
    for date in {current_time.strftime('%Y%m%d'), start_time.strftime('%Y%m%d')}:
        prefix = f"mrms_{date}"
        for obj in s3.Bucket(mrms_bucket).objects.filter(Prefix=prefix):
            try:
                # keys like: mrms_YYYYmmddTHH:MMZ_<field>.pkl.lz4
                file_time = datetime.strptime(obj.key.split('_')[1], '%Y%m%dT%H:%MZ')
                if start_time <= file_time <= current_time:
                    files.append(obj.key)
            except Exception:
                continue

    files = sorted(files, reverse=True)

    if REDIS_AVAILABLE and files:
        r_cache.setex("mrms_file_list", 300, pickle.dumps(files))
        r_cache.setex("mrms_file_list_timestamp", 300, str(time.time()))

    return files

def dilate_mesh(data: np.ndarray, dilation_size=10, threshold=20) -> np.ndarray:
    """Binary-dilate the >= threshold mask and keep original values inside the dilated mask."""
    hail_mask = data >= threshold
    structure = np.ones((2 * dilation_size + 1, 2 * dilation_size + 1), dtype=bool)
    dilated_mask = binary_dilation(hail_mask, structure=structure)
    return np.where(dilated_mask, data, 0.0)

def _resample_to_target(field: np.ndarray, order: int) -> np.ndarray:
    """Resample a 2D field to TARGET_H x TARGET_W with given interpolation order."""
    if field.shape == (TARGET_H, TARGET_W):
        return field
    zoom = [TARGET_H / field.shape[0], TARGET_W / field.shape[1]]
    return scipy.ndimage.zoom(field, zoom, order=order)

# ----------------- Main cycle -----------------
def build_dataset():
    """Build dataset with caching; compute and write ground truth swaths once per timestamp."""
    cycle_start = time.time()

    logger.info("=" * 60)
    logger.info("Starting data build cycle")

    # Pre-allocate output tensor: 12 timesteps x H x W x channels
    x_data = np.zeros((12, TARGET_H, TARGET_W, len(mrms_keys)), dtype=np.float32)

    # Get the list of MRMS files (most recent first)
    all_files = get_mrms_files()
    field_files = all_files[:12]

    if not field_files:
        logger.error("No MRMS files found")
        return None

    # Determine which timesteps we need to download/process (cache miss)
    files_to_download = []
    cache_hits = 0

    for time_idx, file_key in enumerate(field_files):
        if REDIS_AVAILABLE:
            cache_key = f"processed_timestep_{file_key}"
            cached = r_cache.get(cache_key)
            if cached:
                try:
                    timestep_data = pickle.loads(lz4.frame.decompress(cached))
                    if timestep_data.shape == (TARGET_H, TARGET_W, len(mrms_keys)):
                        x_data[time_idx] = timestep_data
                        cache_hits += 1
                        logger.info(f"Cache hit: {file_key}")
                        continue
                except Exception:
                    pass
        files_to_download.append(file_key)

    logger.info(f"Cache: {cache_hits}/{len(field_files)} hits")

    # Download new files (cache misses) in parallel
    file_data = {}
    if files_to_download:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(download_data, mrms_bucket, f): f for f in files_to_download}
            for future in as_completed(futures):
                file_key = futures[future]
                try:
                    file_data[file_key] = future.result()
                except Exception as e:
                    logger.error(f"Failed to download {file_key}: {e}")

    # Build a 60-min mesh cache (may require up to ~30 2-min slices)
    mesh_cache = {}
    for file in all_files[:30]:  # ~60 minutes
        try:
            if file in file_data:
                mesh = file_data[file].get('MESH')
            else:
                mesh = download_data(mrms_bucket, file).get('MESH')

            if mesh is not None:
                mesh = _resample_to_target(mesh, order=1)
                time_str = file.split('_')[1]
                file_time = datetime.strptime(time_str, '%Y%m%dT%H:%MZ')
                mesh_cache[file_time] = mesh
        except Exception:
            continue

    # Compute and write GT for EACH of the 12 target timesteps (deduped)
    written_gt = set()
    mm_by_time = {}  # stash to reuse when building channels
    for file_key in field_files:
        current_time_str = file_key.split('_')[1]
        current_time = datetime.strptime(current_time_str, '%Y%m%dT%H:%MZ')
        stamp_no_colon = current_time.strftime('%Y%m%dT%H%MZ')

        lookback_start = current_time - timedelta(minutes=60)
        mesh_in_window = [m for t, m in mesh_cache.items() if lookback_start <= t <= current_time]

        if mesh_in_window:
            mesh_max_60min = np.maximum.reduce(mesh_in_window)
        else:
            mesh_max_60min = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)

        mm_by_time[current_time] = mesh_max_60min

        gt_key = f"ground_truth/mrms_{stamp_no_colon}/mesh_max_60min.pkl"

        if stamp_no_colon in written_gt:
            continue
        if s3_key_exists(output_bucket, gt_key):
            # already written in a previous cycle
            written_gt.add(stamp_no_colon)
            continue

        try:
            s3_client.put_object(
                Bucket=output_bucket,
                Key=gt_key,
                Body=pickle.dumps(mesh_max_60min, protocol=pickle.HIGHEST_PROTOCOL)
            )
            logger.info(f"Saved ground truth: {gt_key}")
            written_gt.add(stamp_no_colon)
        except Exception as e:
            logger.error(f"Failed to save ground truth {gt_key}: {e}")

    # Process timesteps that were not cache hits
    for file_key in files_to_download:
        if file_key not in file_data:
            continue

        time_idx = field_files.index(file_key)
        mrms_blob = file_data[file_key]

        current_time_str = file_key.split('_')[1]
        current_time = datetime.strptime(current_time_str, '%Y%m%dT%H:%MZ')

        # Fetch the already-computed swath for this timestamp
        mesh_max_60min = mm_by_time.get(current_time)
        if mesh_max_60min is None:
            # Defensive fallback (shouldn't happen)
            lookback_start = current_time - timedelta(minutes=60)
            mesh_in_window = [m for t, m in mesh_cache.items() if lookback_start <= t <= current_time]
            mesh_max_60min = np.maximum.reduce(mesh_in_window) if mesh_in_window else np.zeros((TARGET_H, TARGET_W), dtype=np.float32)

        # Build the channel stack
        timestep_data = np.zeros((TARGET_H, TARGET_W, len(mrms_keys)), dtype=np.float32)

        for field_idx, key in enumerate(mrms_keys):
            field_name = key[5:]  # drop 'mrms_'

            if field_name == 'MESH_Max_60min':
                timestep_data[:, :, field_idx] = scale_data_mrms(mesh_max_60min, field_name)
            # channel 1: raw instantaneous MESH (unscaled, clamped >= 0)
            elif field_name == 'MESH' and field_idx == 1:
                mesh_field = mrms_blob.get('MESH')
                if mesh_field is None: continue
                mesh_field = _resample_to_target(mesh_field, order=2)
                timestep_data[:, :, field_idx] = scale_data_mrms(mesh_field, 'MESH', channel_idx=1)

            # channel 7: dilated MESH (SCALED)
            elif field_name == 'MESH' and field_idx == 7:
                mesh_field = mrms_blob.get('MESH')
                if mesh_field is None: continue
                mesh_field = _resample_to_target(mesh_field, order=2)
                mesh_field = dilate_mesh(mesh_field)
                timestep_data[:, :, field_idx] = scale_data_mrms(mesh_field, 'MESH', channel_idx=7)

            elif field_name in mrms_blob:
                field = mrms_blob[field_name]
                field = _resample_to_target(field, order=2)
                timestep_data[:, :, field_idx] = scale_data_mrms(field, field_name, channel_idx=field_idx)

            else:
                # leave zeros for missing fields
                continue

        x_data[time_idx] = timestep_data

        # Cache processed timestep
        if REDIS_AVAILABLE:
            try:
                cache_key = f"processed_timestep_{file_key}"
                compressed = lz4.frame.compress(pickle.dumps(timestep_data, protocol=pickle.HIGHEST_PROTOCOL))
                r_cache.setex(cache_key, 5400, compressed)
            except Exception:
                pass

    cycle_time = time.time() - cycle_start
    logger.info(f"Cycle complete: {cycle_time:.1f}s, cache rate: {cache_hits}/{len(field_files)}")

    # SAFE channel stats with shape banner
    try:
        shape = getattr(x_data, "shape", None)
        logger.info(f"Post-cycle tensor shape: {shape}")
        C = int(shape[-1]) if shape and len(shape) >= 1 else 0
        for c in range(C):
            arrc = x_data[..., c]
            cmin = float(np.nanmin(arrc)) if arrc.size else float("nan")
            cmax = float(np.nanmax(arrc)) if arrc.size else float("nan")
            logger.info(f"Channel {c}: min {cmin}, max {cmax}")
    except Exception:
        logger.exception("Failed while logging channel ranges")


    return {
        "data": x_data,
        "image_keys_single": mrms_keys,
        "mrms_file": field_files[0] if field_files else "",
        "lats": np.arange(54.98, 19.98, -0.02),
        "lons": np.arange(-130, -60, 0.02),
    }

# ----------------- Runner -----------------
if __name__ == "__main__":
    while True:
        try:
            _ = build_dataset()
        except Exception as e:
            logger.exception(f"Error during cycle: {e}")
        time.sleep(120)  # ~2 minutes
