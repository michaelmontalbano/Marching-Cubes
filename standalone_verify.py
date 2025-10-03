#!/usr/bin/env python3
"""Standalone MESH verification using on-demand MRMS downloads."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evaluation.config import StandaloneConfig
from evaluation.data import S3ArtifactLoader
from evaluation.models import ModelLoader, ModelType
from evaluation.plotting import plot_ground_truth_prediction_difference
from evaluation.mrms import MRMSConfig, MRMSDataBuilder, NormalizationArrays


THRESHOLDS = np.array([5, 10, 20, 30, 40, 50, 70], dtype=float)
SUGGESTED_DATES = (
    "2025-05-28T22:00Z",
    "2025-05-27T18:00Z",
    "2025-05-26T20:20Z",
    "2025-04-10T23:00Z",
)


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("Datetime string cannot be empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise argparse.ArgumentTypeError(f"Unable to parse datetime '{value}'") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ensure_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint to evaluate")
    parser.add_argument("--model_type", choices=["convgru", "flow"], help="Explicitly select the model family")
    parser.add_argument("--datetime", dest="datetimes", action="append", required=True,
                        help="Target datetime (ISO8601, repeatable)")
    parser.add_argument("--lead_time", type=_ensure_positive_int, default=60,
                        help="Lead time in minutes to verify (default: 60)")
    parser.add_argument("--flow_steps", type=_ensure_positive_int, default=64,
                        help="Euler integration steps for diffusion flow models")
    parser.add_argument("--tile_size", type=_ensure_positive_int, default=256,
                        help="Tile size for sliding-window inference")
    parser.add_argument("--stride", type=_ensure_positive_int, default=128,
                        help="Stride for sliding-window inference")
    parser.add_argument("--mrms_bucket", default="noaa-mrms-pds", help="S3 bucket containing MRMS data")
    parser.add_argument("--model_bucket", default="dev-grib-bucket", help="Bucket for normalization arrays")
    parser.add_argument("--norm_min_key", default="global_mins.npy", help="Key for global minima array")
    parser.add_argument("--norm_max_key", default="global_maxs.npy", help="Key for global maxima array")
    parser.add_argument("--cache_dir", default="./cache", help="Local cache directory for S3 downloads")
    parser.add_argument("--output_dir", default="./standalone_outputs",
                        help="Directory where artifacts (plots, npy, json) are written")
    parser.add_argument("--no_plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--save_outputs", action="store_true", help="Persist prediction/ground-truth arrays to disk")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    return parser


def _load_local_array(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            logging.getLogger(__name__).warning("Failed to read local array %s", path)
    return None


def load_normalization_arrays(
    loader: S3ArtifactLoader,
    bucket: str,
    min_key: str,
    max_key: str,
) -> NormalizationArrays:
    min_local = Path(Path(min_key).name)
    max_local = Path(Path(max_key).name)
    global_min = _load_local_array(min_local)
    if global_min is None:
        logging.getLogger(__name__).info("Downloading normalization minima from s3://%s/%s", bucket, min_key)
        arr = loader.load_numpy(bucket, min_key)
        if arr is None:
            raise RuntimeError(f"Unable to load normalization minima from s3://{bucket}/{min_key}")
        np.save(min_local, arr)
        global_min = arr
    global_max = _load_local_array(max_local)
    if global_max is None:
        logging.getLogger(__name__).info("Downloading normalization maxima from s3://%s/%s", bucket, max_key)
        arr = loader.load_numpy(bucket, max_key)
        if arr is None:
            raise RuntimeError(f"Unable to load normalization maxima from s3://{bucket}/{max_key}")
        np.save(max_local, arr)
        global_max = arr
    return NormalizationArrays(global_min=np.asarray(global_min, dtype=np.float32),
                               global_max=np.asarray(global_max, dtype=np.float32))


@dataclass
class PredictionSummary:
    prediction: np.ndarray
    ground_truth: np.ndarray
    metrics: Dict[float, Dict[str, float]]
    plot_path: Optional[str]


class TiledPredictor:
    """Run tiled inference across the CONUS domain."""

    def __init__(
        self,
        model,
        model_type: ModelType,
        tile_size: int,
        stride: int,
        flow_steps: int,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.tile_size = tile_size
        self.stride = stride
        self.flow_steps = flow_steps

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        timesteps, height, width, _ = tensor.shape
        predictions = np.zeros((timesteps, height, width), dtype=np.float32)
        counts = np.zeros((timesteps, height, width), dtype=np.float32)
        tiles_processed = 0

        for y in range(0, height - self.tile_size + 1, self.stride):
            for x in range(0, width - self.tile_size + 1, self.stride):
                tile = tensor[:, y : y + self.tile_size, x : x + self.tile_size, :]
                sequence = self._predict_tile(tile)
                predictions[:, y : y + self.tile_size, x : x + self.tile_size] += sequence
                counts[:, y : y + self.tile_size, x : x + self.tile_size] += 1
                tiles_processed += 1
                if tiles_processed % 100 == 0:
                    logging.getLogger(__name__).info("Processed %d tiles", tiles_processed)

        mask = counts > 0
        predictions[mask] /= counts[mask]
        predictions[predictions < 0.5] = 0.0
        uncovered = np.count_nonzero(counts[0] == 0)
        if uncovered:
            logging.getLogger(__name__).warning("%d pixels were not covered by any tile", uncovered)
        logging.getLogger(__name__).info("Processed %d total tiles", tiles_processed)
        return predictions

    def _predict_tile(self, tile: np.ndarray) -> np.ndarray:
        if self.model_type is ModelType.FLOW:
            return self._predict_flow_tile(tile)
        return self._predict_convgru_tile(tile)

    def _predict_convgru_tile(self, tile: np.ndarray) -> np.ndarray:
        batch = np.expand_dims(tile, axis=0)
        outputs = self.model.predict(batch, verbose=0)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        array = np.asarray(outputs, dtype=np.float32)
        if array.ndim == 5:
            array = array[0, ..., 0]
        elif array.ndim == 4:
            array = array[0]
        return array

    def _predict_flow_tile(self, tile: np.ndarray) -> np.ndarray:
        condition = np.expand_dims(tile.astype(np.float32), axis=0)
        frames: List[np.ndarray] = []
        n_steps = max(1, int(self.flow_steps))
        dt = 1.0 / float(n_steps)
        times = np.linspace(0.0, 1.0 - dt, n_steps, dtype=np.float32)

        for timestep_idx in range(tile.shape[0]):
            state = np.zeros((1, tile.shape[1], tile.shape[2], 1), dtype=np.float32)
            timestep_value = np.array([float(timestep_idx)], dtype=np.float32)
            for t_val in times:
                inputs = {
                    "x_t": state,
                    "condition": condition,
                    "t": np.array([t_val], dtype=np.float32),
                    "timestep_idx": timestep_value,
                }
                velocity = self.model.predict(inputs, verbose=0)
                if isinstance(velocity, (list, tuple)):
                    velocity = velocity[0]
                state = state + velocity.astype(np.float32) * dt
            frames.append(state[0, ..., 0])
        return np.stack(frames, axis=0)


def compute_threshold_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[float, Dict[str, float]]:
    summary: Dict[float, Dict[str, float]] = {}
    for threshold in THRESHOLDS:
        pred_binary = prediction >= threshold
        gt_binary = ground_truth >= threshold
        hits = float(np.sum(pred_binary & gt_binary))
        false_alarms = float(np.sum(pred_binary & ~gt_binary))
        misses = float(np.sum(~pred_binary & gt_binary))
        denom = max(hits + false_alarms + misses, 1.0)
        csi = hits / denom
        pod = hits / max(hits + misses, 1.0)
        far = false_alarms / max(hits + false_alarms, 1.0)
        bias = (hits + false_alarms) / max(hits + misses, 1.0)
        summary[threshold] = {
            "csi": csi,
            "pod": pod,
            "far": far,
            "bias": bias,
            "tp": hits,
            "fp": false_alarms,
            "fn": misses,
        }
    return summary


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _print_metrics(metrics: Dict[float, Dict[str, float]]) -> None:
    print(f"\n{'Threshold':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8} {'Hits':>10} {'FAs':>10}")
    print("-" * 80)
    for threshold in THRESHOLDS:
        record = metrics[threshold]
        print(
            f"{int(threshold):>10} mm "
            f"{record['csi']:>8.3f} {record['pod']:>8.3f} {record['far']:>8.3f} {record['bias']:>8.3f} "
            f"{int(record['tp']):>10} {int(record['fp']):>10}"
        )


def run_verification(
    target_dt: datetime,
    lead_time: int,
    builder: MRMSDataBuilder,
    normalization: NormalizationArrays,
    predictor: TiledPredictor,
    output_dir: Path,
    plot: bool,
) -> PredictionSummary:
    if lead_time % 5 != 0:
        raise ValueError("Lead time must be a multiple of 5 minutes")
    input_tensor = builder.build_input_tensor(target_dt, normalization)
    prediction_sequence = predictor.predict(input_tensor)

    lead_index = lead_time // 5 - 1
    if lead_index < 0 or lead_index >= prediction_sequence.shape[0]:
        raise ValueError(f"Lead time {lead_time} minutes is outside available range")
    prediction = prediction_sequence[lead_index]

    ground_truth_time = target_dt + timedelta(minutes=lead_time)
    ground_truth = builder.build_ground_truth(ground_truth_time)

    metrics = compute_threshold_metrics(prediction, ground_truth)
    _print_metrics(metrics)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Prediction - Min: {prediction.min():.2f}, Max: {prediction.max():.2f}, Mean: {prediction.mean():.3f}")
    print(f"Ground Truth - Min: {ground_truth.min():.2f}, Max: {ground_truth.max():.2f}, Mean: {ground_truth.mean():.3f}")
    print(f"Non-zero pixels - Prediction: {(prediction > 0).sum()}, Ground Truth: {(ground_truth > 0).sum()}")

    plot_path: Optional[str] = None
    if plot:
        _ensure_directory(output_dir)
        filename = f"verify_{target_dt.strftime('%Y%m%d_%H%M')}_lead{lead_time:03d}.png"
        plot_path = plot_ground_truth_prediction_difference(
            ground_truth=ground_truth,
            prediction=prediction,
            title=f"{target_dt.isoformat()} (+{lead_time}m)",
            output_dir=str(output_dir),
            filename=filename,
        )
        if plot_path:
            logging.getLogger(__name__).info("Wrote comparison plot to %s", plot_path)

    return PredictionSummary(prediction=prediction, ground_truth=ground_truth, metrics=metrics, plot_path=plot_path)


def save_outputs(summary: PredictionSummary, target_dt: datetime, lead_time: int, output_dir: Path) -> None:
    _ensure_directory(output_dir)
    prefix = output_dir / f"verify_{target_dt.strftime('%Y%m%d_%H%M')}_lead{lead_time:03d}"
    np.save(f"{prefix}_prediction.npy", summary.prediction)
    np.save(f"{prefix}_ground_truth.npy", summary.ground_truth)
    with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"thresholds": summary.metrics}, handle, indent=2)
    logging.getLogger(__name__).info("Saved outputs to %s", prefix)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    datetimes = [_parse_datetime(value) for value in args.datetimes]
    loader = S3ArtifactLoader(cache_dir=args.cache_dir)
    normalization = load_normalization_arrays(loader, args.model_bucket, args.norm_min_key, args.norm_max_key)

    mrms_config = MRMSConfig(
        bucket=args.mrms_bucket,
        tile_size=args.tile_size,
        stride=args.stride,
    )
    builder = MRMSDataBuilder(mrms_config)

    config = StandaloneConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        flow_steps=args.flow_steps,
    )
    loaded = ModelLoader().load(config)
    predictor = TiledPredictor(
        loaded.model,
        loaded.model_type,
        tile_size=args.tile_size,
        stride=args.stride,
        flow_steps=args.flow_steps,
    )

    output_dir = Path(args.output_dir)
    for dt in datetimes:
        logging.info("=" * 80)
        logging.info("MESH FORECAST VERIFICATION FOR %s", dt.isoformat())
        logging.info("Lead time: %d minutes", args.lead_time)
        summary = run_verification(
            dt,
            args.lead_time,
            builder,
            normalization,
            predictor,
            output_dir,
            plot=not args.no_plots,
        )
        if args.save_outputs:
            save_outputs(summary, dt, args.lead_time, output_dir)

    print("\n" + "=" * 60)
    print("SUGGESTED TEST DATES:")
    print("-" * 60)
    for suggestion in SUGGESTED_DATES:
        print(f"  python {Path(__file__).name} --datetime {suggestion} --model_path {args.model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

