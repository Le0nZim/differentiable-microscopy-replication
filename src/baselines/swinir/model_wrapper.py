"""SwinIR vendor adapter — paper settings override vendor defaults where specified."""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parents[3] / "SwinIR"


def vendor_root() -> Path:
    return VENDOR_ROOT


def ensure_vendor_on_path() -> Path:
    root = vendor_root()
    if not root.exists():
        raise FileNotFoundError(f"Vendor SwinIR not found at {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def import_swinir():
    root = ensure_vendor_on_path()
    try:
        import importlib.util

        module_path = root / "models" / "network_swinir.py"
        spec = importlib.util.spec_from_file_location("vendor_swinir_network", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load SwinIR from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SwinIR
    except ModuleNotFoundError as exc:
        if "timm" in str(exc):
            raise ModuleNotFoundError(
                "timm is required for vendor SwinIR. Install: pip install timm"
            ) from exc
        raise


def build_swinir_from_config(cfg: dict):
    """Build SwinIR model; paper architecture mostly VENDOR_DEFAULT (paper cites [26])."""
    SwinIR = import_swinir()
    return SwinIR(
        upscale=cfg.get("upscale", 1),
        in_chans=cfg.get("in_chans", 1),
        img_size=cfg.get("img_size", 64),
        window_size=cfg.get("window_size", 8),
        img_range=cfg.get("img_range", 1.0),
        depths=cfg.get("depths", [6, 6, 6, 6, 6, 6]),
        embed_dim=cfg.get("embed_dim", 180),
        num_heads=cfg.get("num_heads", [6, 6, 6, 6, 6, 6]),
        mlp_ratio=cfg.get("mlp_ratio", 2),
        upsampler=cfg.get("upsampler", ""),
        resi_connection=cfg.get("resi_connection", "1conv"),
        # Gradient checkpointing trades compute for memory; lets SwinIR-M (embed_dim 180)
        # fit at 256x256 for the MCF7 Fig 8/9 end-to-end training. Default False keeps all
        # existing callers (Table-2/Fig-7, Fig-3) byte-for-byte unchanged.
        use_checkpoint=cfg.get("use_checkpoint", False),
    )
