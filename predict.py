"""Entry point for running predictions with the Kuantum model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from kuantum.models import ModelArtifacts, load_model_and_scaler
from kuantum.simulation.main import run_cli as run_simulation


@dataclass
class Predictor:
    artifacts: ModelArtifacts

    @property
    def device(self) -> torch.device:
        return self.artifacts.device

    @property
    def model(self):
        return self.artifacts.model

    @property
    def scaler(self):
        return self.artifacts.scaler

    def predict(self, features: Sequence[float]) -> int:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        scaled = self.scaler.transform(array)
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        x_categ = torch.empty((tensor.size(0), 0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            x_cont_enc = self.model.encode_continuous(tensor)
            x_categ_enc = self.model.encode_categorical(x_categ)
            logits = self.model(x_categ, tensor, x_categ_enc, x_cont_enc)
            pred = torch.argmax(logits, dim=-1)
        return int(pred.item())


def load_predictor(config_path: Optional[Path] = None) -> Predictor:
    artifacts = load_model_and_scaler(config_path)
    return Predictor(artifacts=artifacts)


def predict_event(features: Sequence[float], config_path: Optional[Path] = None) -> int:
    predictor = load_predictor(config_path)
    return predictor.predict(features)


if __name__ == "__main__":
    run_simulation()
