"""Model loading utilities."""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import tensorflow as tf
from tensorflow import keras as tfk

from . import config as cfg

try:  # Optional dependency used only for legacy checkpoints
    import h5py
except Exception:  # pragma: no cover - environments without h5py
    h5py = None


class _LegacyConv2DTranspose(tfk.layers.Conv2DTranspose):
    """Conv2DTranspose variant that tolerates legacy HDF5 configs."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("groups", None)
        return super().from_config(config)


_LambdaLayer = tfk.layers.Lambda
_original_lambda_compute_output_shape = _LambdaLayer.compute_output_shape


def _safe_lambda_compute_output_shape(self, input_shape):
    """Best-effort Lambda shape inference for legacy H5 checkpoints."""

    try:
        return _original_lambda_compute_output_shape(self, input_shape)
    except NotImplementedError:
        # Fallback heuristics mimic the common identity/reshape lambdas used in
        # the historic ConvGRU exporter where the batch axis is unchanged.
        if isinstance(input_shape, (list, tuple)):
            if not input_shape:
                return input_shape
            # Single input stored in a tuple/list.
            if len(input_shape) == 1:
                return input_shape[0]
            return input_shape[0]
        return input_shape


if getattr(_LambdaLayer.compute_output_shape, "__name__", "") != _safe_lambda_compute_output_shape.__name__:
    _LambdaLayer.compute_output_shape = _safe_lambda_compute_output_shape


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
        model = self._load_with_fallbacks(model_path, custom_objects)
        logger.info("Loaded %s model from %s", model_type.value, model_path)
        return LoadedModel(model=model, model_type=model_type)

    @staticmethod
    def _load_with_fallbacks(path: str, custom_objects: Dict[str, object]) -> tfk.Model:
        try:
            return tfk.models.load_model(path, custom_objects=custom_objects, compile=False)
        except ValueError as exc:
            message = str(exc)
            if (
                "accessible `.keras` zip file" in message
                and h5py is not None
                and h5py.is_hdf5(path)
            ):
                logger.info("Detected legacy HDF5 checkpoint stored with .keras suffix; reloading via temporary .h5 copy")
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                    temp_path = tmp.name
                try:
                    shutil.copy2(path, temp_path)
                    return tfk.models.load_model(temp_path, custom_objects=custom_objects, compile=False)
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            raise
