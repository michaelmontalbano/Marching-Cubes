#!/usr/bin/env python3
"""Evaluate a model across the mesh test set using CSI metrics."""
from __future__ import annotations

import argparse
import logging
from typing import Optional

from evaluation.config import MeshEvaluationConfig
from evaluation.data import EvaluationDataRepository, NormalizationBundle, S3ArtifactLoader
from evaluation.mesh_runner import MeshEvaluator
from evaluation.models import ModelLoader


def parse_n_tiles(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "all", "everything", "full", "max", "none", "*"}:
        return None
    try:
        parsed = int(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--n_tiles must be a positive integer or 'everything'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--n_tiles must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint to evaluate")
    parser.add_argument("--model_type", choices=["convgru", "flow"], help="Explicitly specify the model type")
    parser.add_argument("--n_tiles", type=parse_n_tiles, default=None,
                        help="Number of tiles to evaluate from the dataframe")
    parser.add_argument("--flow_steps", type=int, default=64,
                        help="Euler integration steps for diffusion models")
    parser.add_argument("--bucket", default="dev-grib-bucket", help="S3 bucket containing evaluation assets")
    parser.add_argument("--test_df_key", default="dataframes/test.csv",
                        help="S3 key for the dataframe listing evaluation tiles")
    parser.add_argument("--norm_min_key", default="global_mins.npy", help="S3 key for global minima array")
    parser.add_argument("--norm_max_key", default="global_maxs.npy", help="S3 key for global maxima array")
    parser.add_argument("--cache_dir", default="./cache", help="Local directory for cached downloads")
    parser.add_argument("--summary_dir", default="./evaluation_results", help="Directory for metrics summaries")
    parser.add_argument("--no_save", action="store_true", help="Disable writing JSON summaries and plots")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    config = MeshEvaluationConfig(
        bucket=args.bucket,
        test_df_key=args.test_df_key,
        norm_min_key=args.norm_min_key,
        norm_max_key=args.norm_max_key,
        model_path=args.model_path,
        model_type=args.model_type,
        n_tiles=args.n_tiles,
        flow_steps=args.flow_steps,
        save_summary=not args.no_save,
        summary_dir=args.summary_dir,
    )

    loader = S3ArtifactLoader(cache_dir=args.cache_dir)
    normalization = NormalizationBundle.from_loader(loader, config.bucket, config.norm_min_key, config.norm_max_key)
    repository = EvaluationDataRepository(loader, config.bucket, config.test_df_key)
    loaded_model = ModelLoader().load(config)
    evaluator = MeshEvaluator(config, repository, normalization, loaded_model)
    result = evaluator.evaluate()

    print(f"Model type: {result['model_type']}")
    print(f"Samples processed: {result['num_samples']}")
    print(f"Successful: {result['successful']}, Failed: {result['failed']}")

    overall = result["metrics"]["overall"]
    print("=" * 60)
    print("Overall CSI summary (best thresholds per observation):")
    print(f"{'Obs Thr':>8} {'Best Pred':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8}")
    for record in overall.best_metrics:
        print(
            f"{record['obs_threshold']:>7.0f}mm {record['best_pred_threshold']:>9.1f}mm "
            f"{record['CSI']:>8.3f} {record['POD']:>8.3f} {record['FAR']:>8.3f} {record['Bias']:>8.3f}"
        )

    key_timesteps = [0, 6, 11]
    for timestep in key_timesteps:
        key = f"timestep_{timestep}"
        if key not in result["metrics"]:
            continue
        summary = result["metrics"][key]
        minutes = timestep * 5
        print("-" * 60)
        print(f"Timestep {timestep} ({minutes} minutes)")
        print(f"{'Obs Thr':>8} {'Best Pred':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8}")
        for record in summary.best_metrics:
            print(
                f"{record['obs_threshold']:>7.0f}mm {record['best_pred_threshold']:>9.1f}mm "
                f"{record['CSI']:>8.3f} {record['POD']:>8.3f} {record['FAR']:>8.3f} {record['Bias']:>8.3f}"
            )


if __name__ == "__main__":
    main()
