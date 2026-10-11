"""Upstream-aligned OCR result serialization and display."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from mlx4ocr.types import BoundingBox, OCRResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRTiming:
    """Elapsed seconds for pipeline stages."""

    det_s: float
    rec_s: float
    total_s: float


def box_points_int32(box: BoundingBox) -> list[list[int]]:
    """Convert a bounding box to Paddle ``predict_system`` int32 points."""
    return [[round(x), round(y)] for x, y in box.points]


def require_paddlex_block_fields(result: OCRResult) -> None:
    """Validate that every block has PaddleX geometry and scores."""
    paddlex_blocks(result)


def paddlex_blocks(result: OCRResult) -> tuple[tuple[BoundingBox, str, float, float], ...]:
    """Return PaddleX-ready block fields after validating required metadata."""
    fields: list[tuple[BoundingBox, str, float, float]] = []
    for index, block in enumerate(result.blocks):
        if block.box is None:
            raise ValueError(f"OCR block {index} is missing geometry required for PaddleX output")
        if block.detection_score is None:
            raise ValueError(
                f"OCR block {index} is missing detection score required for PaddleX output"
            )
        if block.recognition_score is None:
            raise ValueError(
                f"OCR block {index} is missing recognition score required for PaddleX output"
            )
        fields.append((block.box, block.text, block.detection_score, block.recognition_score))
    return tuple(fields)


def to_system_results_entries(result: OCRResult) -> list[dict[str, object]]:
    """Build Paddle ``predict_system`` JSON entries.

    Args:
        result: Filtered OCR output with blocks containing geometry.

    Returns:
        List of ``{"transcription": str, "points": [[x,y], ...]}`` dicts.
    """
    entries: list[dict[str, object]] = []
    for box, text, _detection_score, _recognition_score in paddlex_blocks(result):
        entries.append(
            {
                "transcription": text,
                "points": box_points_int32(box),
            }
        )
    return entries


def to_system_results_line(result: OCRResult, basename: str) -> str:
    """Format one ``system_results.txt`` line for Paddle parity.

    Args:
        result: OCR output for a single image.
        basename: Image file name without directory.

    Returns:
        TSV line ``{basename}\\t{json}\\n``.
    """
    payload = to_system_results_entries(result)
    return f"{basename}\t{json.dumps(payload, ensure_ascii=False)}\n"


def to_markdown(result: OCRResult) -> str:
    """Format OCR output as Markdown body text.

    Args:
        result: OCR output for a single image.

    Returns:
        Recognized text with one trailing newline when non-empty.
    """
    if not result.text:
        return ""
    return result.text + "\n"


def save_to_markdown(
    result: OCRResult,
    save_path: Path,
    *,
    input_path: str | None = None,
) -> Path:
    """Save OCR output as Markdown.

    Args:
        result: OCR output for a single image.
        save_path: Target file or directory. Directories receive ``{stem}.md``.
        input_path: Optional source image path recorded in Markdown.

    Returns:
        Path to the written Markdown file.
    """
    if save_path.suffix != ".md":
        stem = Path(input_path or "result").stem
        save_path.mkdir(parents=True, exist_ok=True)
        markdown_path = save_path / f"{stem}.md"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path = save_path

    markdown_path.write_text(
        to_markdown(result),
        encoding="utf-8",
    )
    logger.info("Wrote %s", markdown_path)
    return markdown_path


def to_paddlex_res(
    result: OCRResult,
    *,
    input_path: str | None = None,
    page_index: int | None = None,
    text_rec_score_thresh: float = 0.0,
    word_boxes: bool = False,
) -> dict[str, object]:
    """Build a PaddleOCR 3.x ``res`` dict from filtered OCR output.

    With ``word_boxes`` the dict also carries PaddleX's ``text_word`` (one list of
    characters per line) and ``text_word_boxes`` (same-shaped ``[x1, y1, x2, y2]``
    cells); the result must have been produced with ``predict(..., char_boxes=True)``.

    Args:
        result: Filtered OCR output with blocks containing geometry and scores.
        input_path: Optional source image path.
        page_index: Optional PDF page index.
        text_rec_score_thresh: Score threshold recorded in metadata.

    Returns:
        Mapping compatible with PaddleX ``save_to_json`` core fields.
    """
    rec_polys: list[list[list[int]]] = []
    rec_texts: list[str] = []
    rec_scores: list[float] = []
    rec_boxes: list[list[float]] = []
    dt_scores: list[float] = []
    for box, text, detection_score, recognition_score in paddlex_blocks(result):
        rec_polys.append(box_points_int32(box))
        rec_texts.append(text)
        rec_scores.append(recognition_score)
        rec_boxes.append([box.x_min, box.y_min, box.x_max, box.y_max])
        dt_scores.append(detection_score)
    res: dict[str, object] = {
        "input_path": input_path,
        "page_index": page_index,
        "dt_polys": rec_polys,
        "dt_scores": dt_scores,
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys,
        "rec_boxes": rec_boxes,
        "text_rec_score_thresh": text_rec_score_thresh,
    }
    if word_boxes:
        for index, block in enumerate(result.blocks):
            if "".join(block.chars) != block.text or len(block.chars) != len(block.char_boxes):
                raise ValueError(f"OCR block {index} has no usable character boxes")
        res["text_word"] = [list(block.chars) for block in result.blocks]
        res["text_word_boxes"] = [
            [[round(value, 2) for value in cell] for cell in block.char_boxes]
            for block in result.blocks
        ]
    return res


def save_to_json(
    result: OCRResult,
    save_path: Path,
    *,
    input_path: str | None = None,
    page_index: int | None = None,
    text_rec_score_thresh: float = 0.0,
) -> Path:
    """Save OCR output as PaddleX-style JSON.

    Args:
        result: OCR output for a single image.
        save_path: Target file or directory. Directories receive
            ``{stem}_res.json``.
        input_path: Optional source image path stored in JSON.
        page_index: Optional PDF page index stored in JSON.
        text_rec_score_thresh: Score threshold recorded in metadata.

    Returns:
        Path to the written JSON file.
    """
    if save_path.suffix != ".json":
        stem = Path(input_path or "result").stem
        save_path.mkdir(parents=True, exist_ok=True)
        json_path = save_path / f"{stem}_res.json"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = save_path

    payload = {
        "res": to_paddlex_res(
            result,
            input_path=input_path,
            page_index=page_index,
            text_rec_score_thresh=text_rec_score_thresh,
        )
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)
    return json_path


def print_result(result: OCRResult) -> None:
    """Print OCR lines in Paddle ``predict_system`` debug style."""
    for _box, text, _detection_score, recognition_score in paddlex_blocks(result):
        print(f"{text}\t{recognition_score:.3f}")
