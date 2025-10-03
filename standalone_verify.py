#!/usr/bin/env python3
"""Run targeted verification for individual tiles using a single model."""
from __future__ import annotations

import argparse
import logging
from typing import Optional

from evaluation.config import StandaloneConfig
from evaluation.data import EvaluationDataRepository, NormalizationBundle, S3ArtifactLoader
from evaluation.models import ModelLoader
from evaluation.standalone_runner import StandaloneVerifier


def parse_n_tiles(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "all", "everything", "full", "max", "none", "*"}:
        return None
    try:
        parsed = int(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--n_tiles must be a positive integer or 'everything'"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--n_tiles must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint to evaluate")
    parser.add_argument("--model_type", choices=["convgru", "flow"], help="Explicitly specify the model type")
    parser.add_argument("--datetime", dest="datetimes", action="append", default=[],
                        help="ISO8601 datetime to evaluate (can be provided multiple times)")
    parser.add_argument("--n_tiles", type=parse_n_tiles, default=None,
                        help="Number of tiles to load after filtering; defaults to all")
    parser.add_argument("--flow_steps", type=int, default=64,
                        help="Euler integration steps for diffusion flow models")
    parser.add_argument("--bucket", default="dev-grib-bucket", help="S3 bucket containing evaluation assets")
    parser.add_argument("--test_df_key", default="dataframes/test.csv",
                        help="S3 key for the dataframe listing evaluation tiles")
    parser.add_argument("--norm_min_key", default="global_mins.npy", help="S3 key for global minima array")
    parser.add_argument("--norm_max_key", default="global_maxs.npy", help="S3 key for global maxima array")
    parser.add_argument("--cache_dir", default="./cache", help="Local directory for cached downloads")
    parser.add_argument("--plot_dir", default="./verification_plots", help="Directory to write comparison plots")
    parser.add_argument("--no_plots", action="store_true", help="Disable writing comparison plots")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    config = StandaloneConfig(
        bucket=args.bucket,
        test_df_key=args.test_df_key,
        norm_min_key=args.norm_min_key,
        norm_max_key=args.norm_max_key,
        model_path=args.model_path,
        model_type=args.model_type,
        n_tiles=args.n_tiles,
        flow_steps=args.flow_steps,
        datetimes=args.datetimes,
        plot_dir=None if args.no_plots else args.plot_dir,
        plot=not args.no_plots,
        save_plots=not args.no_plots,
    )

    loader = S3ArtifactLoader(cache_dir=args.cache_dir)
    normalization = NormalizationBundle.from_loader(loader, config.bucket, config.norm_min_key, config.norm_max_key)
    repository = EvaluationDataRepository(loader, config.bucket, config.test_df_key)
    loaded_model = ModelLoader().load(config)
    verifier = StandaloneVerifier(config, repository, normalization, loaded_model)
    result = verifier.evaluate()

    print(f"Model type: {result['model_type']}")
    print(f"Evaluated samples: {result['num_samples']}")

    evaluations = result["evaluations"]
    for sample in evaluations:
        print("-" * 60)
        print(f"Tile index: {sample.index}")
        if sample.metadata:
            for key, value in sample.metadata.items():
                print(f"  {key}: {value}")
        print(f"  Plot: {sample.plot_path}")
        final_frame = sample.prediction.final_frame.squeeze()
        print(f"  Prediction stats -> min: {final_frame.min():.3f}, max: {final_frame.max():.3f}, mean: {final_frame.mean():.3f}")

    metrics = result["metrics"]
    overall = metrics["overall"]
    print("=" * 60)
    print("Overall CSI summary (best thresholds per observation):")
    print(f"{'Obs Thr':>8} {'Best Pred':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8}")
    for record in overall.best_metrics:
        print(
            f"{record['obs_threshold']:>7.0f}mm {record['best_pred_threshold']:>9.1f}mm "
            f"{record['CSI']:>8.3f} {record['POD']:>8.3f} {record['FAR']:>8.3f} {record['Bias']:>8.3f}"
        )


if __name__ == "__main__":
    main()
