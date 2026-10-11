"""CTC recognition post-processing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mlx4ocr.types import TextRecognition


def load_character_dict(
    dict_path: Path,
    *,
    use_space_char: bool = True,
) -> tuple[str, ...]:
    """Load a PP-OCR character dictionary from disk.

    Args:
        dict_path: Path to a newline-delimited character file.
        use_space_char: Whether to append a space character to the dictionary.

    Returns:
        Character list with the CTC blank token prepended.
    """
    if not dict_path.is_file():
        raise FileNotFoundError(f"missing character dictionary: {dict_path}")

    characters: list[str] = []
    with dict_path.open(encoding="utf-8") as handle:
        for line in handle:
            characters.append(line.strip("\n").strip("\r\n"))
    if use_space_char:
        characters.append(" ")
    return ("blank", *characters)


def _pred_reverse(pred: str) -> str:
    pred_parts: list[str] = []
    current = ""
    for char in pred:
        if not re.search(r"[a-zA-Z0-9 :*./%+-]", char):
            if current:
                pred_parts.append(current)
            pred_parts.append(char)
            current = ""
        else:
            current += char
    if current:
        pred_parts.append(current)
    return "".join(pred_parts[::-1])


def ctc_decode(
    preds: np.ndarray,
    characters: Sequence[str],
    *,
    reverse: bool = False,
    valid_ratios: Sequence[float] | None = None,
) -> tuple[TextRecognition, ...]:
    """Decode CTC softmax outputs into text labels.

    Args:
        preds: Softmax tensor ``[B, T, num_classes]``.
        characters: Character table with blank at index 0.
        reverse: Whether to reverse mixed Arabic-like predictions.
        valid_ratios: Per batch item, the fraction of the ``T`` time steps that
            covers the crop (the rest is right padding). When given, each result
            carries ``chars`` / ``char_centers`` / ``char_width`` expressed as
            fractions of the crop's reading axis. Ignored when ``reverse`` is set,
            because reversal breaks the char-to-position order.

    Returns:
        One ``TextRecognition`` per batch item.
    """
    if preds.ndim != 3:
        raise ValueError(f"expected softmax [B, T, C], got shape {preds.shape}")
    if valid_ratios is not None and len(valid_ratios) != preds.shape[0]:
        raise ValueError("valid_ratios must have one entry per batch item")

    steps = preds.shape[1]
    indices = preds.argmax(axis=2)
    probs = preds.max(axis=2)
    results: list[TextRecognition] = []
    for batch_idx in range(indices.shape[0]):
        sequence = indices[batch_idx]
        confidence = probs[batch_idx]
        selection = np.ones(sequence.shape[0], dtype=bool)
        selection[1:] = sequence[1:] != sequence[:-1]
        selection &= sequence != 0

        char_list = [characters[int(text_id)] for text_id in sequence[selection]]
        conf_list = confidence[selection]
        if conf_list.size == 0:
            score = 0.0
        else:
            score = float(np.mean(conf_list))

        text = "".join(char_list)
        if reverse:
            text = _pred_reverse(text)
        if valid_ratios is None or reverse:
            results.append(TextRecognition(text=text, score=score))
            continue

        span = steps * valid_ratios[batch_idx]
        peaks = np.flatnonzero(selection)
        centers = np.clip((peaks + 0.5) / span, 0.0, 1.0)
        results.append(
            TextRecognition(
                text=text,
                score=score,
                chars=tuple(char_list),
                char_centers=tuple(float(center) for center in centers),
                char_width=1.0 / span,
            )
        )
    return tuple(results)
