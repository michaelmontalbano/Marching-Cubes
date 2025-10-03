"""Model loading utilities."""
from __future__ import annotations

import importlib
import inspect
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import tensorflow as tf
from tensorflow import keras as tfk

from . import config as cfg


class _LegacyConv2DTranspose(tfk.layers.Conv2DTranspose):
    """Conv2DTranspose variant that tolerates legacy HDF5 configs."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def _fallback_weighted_mse():
    """Return a minimal weighted MSE implementation for ConvGRU checkpoints."""

    def loss(y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        timesteps = tf.shape(y_true)[1]
        timestep_weights = tf.range(1, timesteps + 1, dtype=tf.float32)
        timestep_weights = tf.reshape(timestep_weights, (1, timesteps, 1, 1, 1))
        return mse * timestep_weights

    return loss


def _fallback_csi(threshold: float = 20.0):
    """Simplified CSI metric used by historic ConvGRU training runs."""

    def metric(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true_binary = tf.cast(y_true > threshold, tf.float32)

        tp = tf.reduce_sum(y_true_binary[:, -1] * y_pred_binary[:, -1])
        fn = tf.reduce_sum(y_true_binary[:, -1] * (1 - y_pred_binary[:, -1]))
        fp = tf.reduce_sum((1 - y_true_binary[:, -1]) * y_pred_binary[:, -1])
        return tp / tf.maximum(tp + fn + fp, 1.0)

    return metric

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    CONVGRU = "convgru"
    FLOW = "flow"

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["ModelType"]:
        if value is None:
            return None
        lowered = value.lower()
        for member in cls:
            if lowered in {member.value, member.name.lower()}:
                return member
        raise ValueError(f"Unknown model type '{value}'")


@dataclass
class LoadedModel:
    model: tfk.Model
    model_type: ModelType


def _load_flow_custom_objects() -> Dict[str, object]:
    try:
        module = importlib.import_module("flow_matching_model")
    except ImportError:
        logger.debug("flow_matching_model module not available")
        return {}
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    objects: Dict[str, object] = {}
    for name in names:
        attr = getattr(module, name)
        if inspect.isfunction(attr) or inspect.isclass(attr):
            objects[name] = attr
    return objects


def _convgru_custom_objects() -> Dict[str, object]:
    try:
        module = importlib.import_module("models")
        weighted_mse = module.weighted_mse
        csi = module.csi
    except Exception as exc:  # pragma: no cover - logging fallback path
        logger.warning("Falling back to bundled ConvGRU losses: %s", exc)
        weighted_mse = _fallback_weighted_mse
        csi = _fallback_csi()
    from rnn import (
        reshape_and_stack,
        slice_to_n_steps,
        slice_output_shape,
        ResBlock,
        WarmUpCosineDecayScheduler,
        ConvGRU,
        ConvBlock,
        ZeroLikeLayer,
        ReflectionPadding2D,
        ResGRU,
        GRUResBlock,
    )

    loss_fn = weighted_mse()
    return {
        "loss": loss_fn,
        "weighted_mse": loss_fn,
        "csi": csi,
        "Conv2DTranspose": _LegacyConv2DTranspose,
        "reshape_and_stack": reshape_and_stack,
        "slice_to_n_steps": slice_to_n_steps,
        "slice_output_shape": slice_output_shape,
        "ResBlock": ResBlock,
        "WarmUpCosineDecayScheduler": WarmUpCosineDecayScheduler,
        "ConvGRU": ConvGRU,
        "ConvBlock": ConvBlock,
        "ZeroLikeLayer": ZeroLikeLayer,
        "ReflectionPadding2D": ReflectionPadding2D,
        "ResGRU": ResGRU,
        "GRUResBlock": GRUResBlock,
    }


def _custom_objects_for(model_type: ModelType) -> Dict[str, object]:
    objects = _convgru_custom_objects()
    if model_type is ModelType.FLOW:
        objects.update(_load_flow_custom_objects())
    return objects


def _guess_model_type(path: str) -> ModelType:
    name = os.path.basename(path).lower()
    if "flow" in name or "diff" in name:
        return ModelType.FLOW
    return ModelType.CONVGRU


class ModelLoader:
    """Load either ConvGRU or flow matching checkpoints."""

    def load(self, config: cfg.EvaluationConfig) -> LoadedModel:
        model_path = config.resolved_model_path()
        if not model_path:
            raise ValueError("A local model path must be supplied via --model_path")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist")

        explicit_type = ModelType.from_string(config.model_type) if config.model_type else None
        model_type = explicit_type or _guess_model_type(model_path)
        custom_objects = _custom_objects_for(model_type)
        model = tfk.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        logger.info("Loaded %s model from %s", model_type.value, model_path)
        return LoadedModel(model=model, model_type=model_type)
