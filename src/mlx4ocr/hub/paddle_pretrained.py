"""Download and load official Paddle pretrained recognition checkpoints."""

from __future__ import annotations

import logging
import pickle
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from mlx4ocr.hub.registry import ModelVariant

logger = logging.getLogger(__name__)

PADDLE_PRETRAINED_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model"
)

_DEFAULT_CACHE_DIR = Path(".cache") / "paddle_pretrained"


def pretrained_rec_url(variant: ModelVariant) -> str:
    """Return the download URL for a PP-OCRv6 recognition pretrained checkpoint.

    Args:
        variant: Model size tier.

    Returns:
        URL to the official ``.pdparams`` file.
    """
    return f"{PADDLE_PRETRAINED_BASE_URL}/PP-OCRv6_{variant}_rec_pretrained.pdparams"


def download_pretrained_rec(
    variant: ModelVariant,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Download and cache an official Paddle pretrained recognition checkpoint.

    Args:
        variant: Model size tier.
        cache_dir: Optional cache directory. Defaults to ``.cache/paddle_pretrained``.

    Returns:
        Path to the cached ``.pdparams`` file.

    Raises:
        FileNotFoundError: If the downloaded file is missing after retrieval.
    """
    root = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"PP-OCRv6_{variant}_rec_pretrained.pdparams"
    if destination.is_file():
        return destination

    url = pretrained_rec_url(variant)
    logger.info("Downloading Paddle pretrained recognition weights from %s", url)
    # Download beside the target and rename, so an interrupted download never leaves a
    # truncated checkpoint that later loads would trip over.
    partial = destination.with_name(destination.name + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)
    if not destination.is_file():
        raise FileNotFoundError(f"failed to download pretrained checkpoint: {destination}")
    return destination


def load_pretrained_rec_state(path: Path) -> dict[str, np.ndarray]:
    """Load a Paddle pretrained recognition checkpoint into numpy arrays.

    Args:
        path: Path to a ``.pdparams`` pickle file.

    Returns:
        Mapping from Paddle parameter names to numpy arrays.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the checkpoint contains no array parameters.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing pretrained checkpoint: {path}")

    raw = pickle.loads(path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError(f"expected dict checkpoint in {path}, got {type(raw).__name__}")

    state: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if hasattr(value, "shape"):
            state[str(key)] = np.asarray(value)

    if not state:
        raise ValueError(f"no array parameters found in pretrained checkpoint: {path}")
    return state


def load_pretrained_rec(
    variant: ModelVariant,
    *,
    cache_dir: Path | None = None,
) -> Mapping[str, np.ndarray]:
    """Download (if needed) and load a pretrained recognition checkpoint.

    Args:
        variant: Model size tier.
        cache_dir: Optional cache directory.

    Returns:
        Paddle pretrained parameter arrays keyed by checkpoint names.
    """
    path = download_pretrained_rec(variant, cache_dir=cache_dir)
    return load_pretrained_rec_state(path)
