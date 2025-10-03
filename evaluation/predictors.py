"""Prediction helpers for different model families."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import EvaluationConfig
from .models import LoadedModel, ModelType

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    sequence: np.ndarray
    final_frame: np.ndarray


class BasePredictor:
    def __init__(self, loaded: LoadedModel, config: EvaluationConfig):
        self.model = loaded.model
        self.model_type = loaded.model_type
        self.config = config

    def predict(self, inputs: np.ndarray) -> PredictionResult:
        raise NotImplementedError


class ConvGRUPredictor(BasePredictor):
    def predict(self, inputs: np.ndarray) -> PredictionResult:
        batch = np.expand_dims(inputs, axis=0)
        outputs = self.model.predict(batch, verbose=0)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        sequence = np.asarray(outputs, dtype=np.float32)
        final_frame = sequence[-1]
        return PredictionResult(sequence=sequence, final_frame=final_frame)


class FlowPredictor(BasePredictor):
    def _integrate_single_timestep(self, condition_batch: np.ndarray, timestep_idx: int) -> np.ndarray:
        height = condition_batch.shape[2]
        width = condition_batch.shape[3]
        state = np.zeros((1, height, width, 1), dtype=np.float32)
        n_steps = max(1, int(self.config.flow_steps))
        dt = 1.0 / float(n_steps)
        times = np.linspace(0.0, 1.0 - dt, n_steps, dtype=np.float32)
        timestep_idx_arr = np.array([float(timestep_idx)], dtype=np.float32)
        for t_val in times:
            t_arr = np.array([t_val], dtype=np.float32)
            inputs = {
                "x_t": state,
                "condition": condition_batch,
                "t": t_arr,
                "timestep_idx": timestep_idx_arr,
            }
            velocity = self.model.predict(inputs, verbose=0)
            if isinstance(velocity, (list, tuple)):
                velocity = velocity[0]
            state = state + velocity.astype(np.float32) * dt
        return state[0]

    def predict(self, inputs: np.ndarray) -> PredictionResult:
        condition = np.expand_dims(inputs.astype(np.float32), axis=0)
        frames = []
        for idx in range(inputs.shape[0]):
            frames.append(self._integrate_single_timestep(condition, idx))
        sequence = np.stack(frames, axis=0)
        final_frame = sequence[-1]
        return PredictionResult(sequence=sequence, final_frame=final_frame)


class PredictorFactory:
    @staticmethod
    def create(loaded: LoadedModel, config: EvaluationConfig) -> BasePredictor:
        if loaded.model_type is ModelType.FLOW:
            return FlowPredictor(loaded, config)
        return ConvGRUPredictor(loaded, config)
