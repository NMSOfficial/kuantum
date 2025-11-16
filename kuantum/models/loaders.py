"""Model and scaler loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pickle

import torch
import yaml

from .tab_attention import TabAttention


@dataclass
class ModelConfig:
    """Configuration structure for a saved model."""

    model_path: Path
    scaler_path: Path
    num_continuous: int
    categories: Tuple[int, ...]
    dim: int
    depth: int
    heads: int
    dim_head: int
    dim_out: int
    attn_dropout: float
    ff_dropout: float
    lastmlp_dropout: float
    cont_embeddings: str
    attentiontype: str
    device: str = "auto"


@dataclass
class ModelArtifacts:
    """Container holding the model, scaler and configuration."""

    model: TabAttention
    scaler: Any
    device: torch.device
    config: ModelConfig


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _parse_config(config: Dict[str, Any], base_path: Path) -> ModelConfig:
    model_cfg = config["model"]
    scaler_cfg = config["scaler"]
    device_cfg = config.get("device", "auto")
    return ModelConfig(
        model_path=base_path / model_cfg["path"],
        scaler_path=base_path / scaler_cfg["path"],
        num_continuous=int(model_cfg["num_continuous"]),
        categories=tuple(model_cfg.get("categories", [])),
        dim=int(model_cfg["dim"]),
        depth=int(model_cfg["depth"]),
        heads=int(model_cfg["heads"]),
        dim_head=int(model_cfg["dim_head"]),
        dim_out=int(model_cfg["dim_out"]),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
        ff_dropout=float(model_cfg.get("ff_dropout", 0.0)),
        lastmlp_dropout=float(model_cfg.get("lastmlp_dropout", 0.0)),
        cont_embeddings=str(model_cfg.get("cont_embeddings", "MLP")),
        attentiontype=str(model_cfg.get("attentiontype", "col")),
        device=str(device_cfg),
    )


def load_config(config_path: Optional[Path] = None) -> ModelConfig:
    raw_config = _load_yaml_config(config_path)
    base_path = (config_path or _DEFAULT_CONFIG_PATH).parent
    return _parse_config(raw_config, base_path)


def _resolve_device(config: ModelConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def load_model(config: Optional[ModelConfig] = None, *, config_path: Optional[Path] = None) -> Tuple[TabAttention, torch.device, ModelConfig]:
    cfg = config or load_config(config_path)
    device = _resolve_device(cfg)

    model = TabAttention(
        categories=cfg.categories,
        num_continuous=cfg.num_continuous,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        dim_out=cfg.dim_out,
        attn_dropout=cfg.attn_dropout,
        ff_dropout=cfg.ff_dropout,
        lastmlp_dropout=cfg.lastmlp_dropout,
        cont_embeddings=cfg.cont_embeddings,
        attentiontype=cfg.attentiontype,
    ).to(device)

    state_dict = torch.load(cfg.model_path, map_location=device)

    # Older checkpoints can store a zero-sized categorical mask embedding when
    # the training run did not use categorical features.  Recent versions of the
    # architecture allocate a dummy slot instead, so we patch the loaded state
    # dict to avoid size mismatches when the stored tensor is empty.
    mask_key = "mask_embeds_cat.weight"
    if mask_key in state_dict:
        weight = state_dict[mask_key]
        if hasattr(weight, "numel") and weight.numel() == 0:
            state_dict[mask_key] = model.mask_embeds_cat.weight.detach().clone()

    model.load_state_dict(state_dict)
    model.eval()
    return model, device, cfg


def load_scaler(config: Optional[ModelConfig] = None, *, config_path: Optional[Path] = None):
    cfg = config or load_config(config_path)
    with open(cfg.scaler_path, "rb") as handle:
        scaler = pickle.load(handle)
    return scaler


def load_model_and_scaler(config_path: Optional[Path] = None) -> ModelArtifacts:
    cfg = load_config(config_path)
    model, device, cfg = load_model(cfg, config_path=config_path)
    scaler = load_scaler(cfg)
    return ModelArtifacts(model=model, scaler=scaler, device=device, config=cfg)
