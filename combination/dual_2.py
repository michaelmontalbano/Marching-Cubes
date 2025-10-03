import time
import boto3
import pickle
import os
import redis
import numpy as np
from dotenv import load_dotenv
from data_combiner import build_dataset
from datetime import datetime
import logging
import zlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dual.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Redis connection
redis_host = os.getenv("REDIS_URL_0", "localhost")
r0 = redis.Redis(host=redis_host, port=6379)
assert r0.ping(), "Cannot connect to Redis"

s3 = boto3.client("s3")

def store_data_chunked(redis_client, key, data, chunk_size_mb=50):
    """Store large data in Redis chunks"""
    compressed_data = zlib.compress(pickle.dumps(data), level=6)
    chunk_size = chunk_size_mb * 1024 * 1024
    num_chunks = (len(compressed_data) + chunk_size - 1) // chunk_size
    
    # Store metadata
    meta = {
        'total_size': len(compressed_data),
        'num_chunks': num_chunks,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'compression': 'zlib'
    }
    redis_client.set(f"{key}_meta", pickle.dumps(meta), ex=7200)
    
    # Store chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(compressed_data))
        redis_client.set(f"{key}_chunk_{i}", compressed_data[start:end], ex=7200)
    
    logger.info(f"Stored {key}: {num_chunks} chunks, {len(compressed_data)} bytes")
    return True

if __name__ == "__main__":
    while True:
        try:
            cycle_start = time.time()
            
            # Build dataset
            data_temp = build_dataset()
            if not data_temp:
                logger.warning("No data available")
                time.sleep(60)
                continue
                
            data = data_temp["data"]
            mrms_file = data_temp["mrms_file"]
            
            # Store data in Redis for predict.py
            store_data_chunked(r0, "data", data)
            
            # Update MRMS file list
            mrms_files_raw = r0.get("mrms_files")
            if mrms_files_raw:
                mrms_files = np.frombuffer(mrms_files_raw, "<U20")
                # Clean up old entries
                if len(mrms_files) > 150:
                    mrms_files = mrms_files[-150:]
            else:
                mrms_files = np.array([], dtype="<U20")
            
            mrms_files = np.append(mrms_files, mrms_file).astype("<U20")
            r0.set("mrms_files", mrms_files.tobytes(), ex=7200)
            
            # Signal ready
            r0.set("isReady", "1", ex=7200)
            
            cycle_time = time.time() - cycle_start
            logger.info(f"Cycle complete: {cycle_time:.1f}s, latest: {mrms_file}")
            
            time.sleep(120)
            
        except Exception as e:
            logger.exception(f"Error in dual_2 loop: {e}")
            time.sleep(120)