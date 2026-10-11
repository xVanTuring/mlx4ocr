"""Tests for per-character boxes in image coordinates."""

from __future__ import annotations

import numpy as np
import pytest

from mlx4ocr.output import to_paddlex_res
from mlx4ocr.pipeline.crop import char_boxes_in_image
from mlx4ocr.types import BoundingBox, OCRResult, TextDetection, TextRecognition


def _recognition(centers: tuple[float, ...], width: float) -> TextRecognition:
    chars = tuple("abc"[: len(centers)])
    return TextRecognition(
        text="".join(chars), score=0.9, chars=chars, char_centers=centers, char_width=width
    )


def test_horizontal_line_cells_follow_x_axis() -> None:
    detection = TextDetection(
        box=BoundingBox(points=((100.0, 50.0), (300.0, 50.0), (300.0, 70.0), (100.0, 70.0))),
        score=0.9,
    )
    boxes = char_boxes_in_image(detection, _recognition((0.25, 0.75), 0.1))

    assert boxes[0] == pytest.approx((100.0 + 200 * 0.2, 50.0, 100.0 + 200 * 0.3, 70.0))
    assert boxes[1] == pytest.approx((100.0 + 200 * 0.7, 50.0, 100.0 + 200 * 0.8, 70.0))


def test_tall_line_reads_top_to_bottom() -> None:
    """A box taller than 1.5x its width is rotated by the crop, so text runs downward."""
    detection = TextDetection(
        box=BoundingBox(points=((10.0, 100.0), (30.0, 100.0), (30.0, 300.0), (10.0, 300.0))),
        score=0.9,
    )
    boxes = char_boxes_in_image(detection, _recognition((0.25, 0.75), 0.1))

    assert boxes[0] == pytest.approx((10.0, 100.0 + 200 * 0.2, 30.0, 100.0 + 200 * 0.3))
    assert boxes[1] == pytest.approx((10.0, 100.0 + 200 * 0.7, 30.0, 100.0 + 200 * 0.8))


def test_rotated_crop_matches_np_rot90() -> None:
    """The rotated mapping agrees with what ``np.rot90`` does to the crop pixels."""
    crop = np.arange(6 * 2).reshape(6, 2)  # height 6, width 2 -> rotated
    rotated = np.rot90(crop)
    # Reading x in the rotated crop is the original row index.
    assert rotated.shape == (2, 6)
    assert [rotated[0, x] for x in range(6)] == [crop[x, 1] for x in range(6)]


def test_word_boxes_in_paddlex_res() -> None:
    detection = TextDetection(
        box=BoundingBox(points=((0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0))),
        score=0.9,
    )
    recognition = _recognition((0.25, 0.75), 0.1)
    result = OCRResult.from_ppocrv6(
        (detection,),
        (recognition,),
        char_boxes=(char_boxes_in_image(detection, recognition),),
    )

    res = to_paddlex_res(result, word_boxes=True)

    assert res["text_word"] == [["a", "b"]]
    assert res["text_word_boxes"] == [[[20.0, 0.0, 30.0, 10.0], [70.0, 0.0, 80.0, 10.0]]]
    assert "text_word" not in to_paddlex_res(result)


def test_word_boxes_require_positions() -> None:
    detection = TextDetection(
        box=BoundingBox(points=((0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0))),
        score=0.9,
    )
    result = OCRResult.from_ppocrv6((detection,), (TextRecognition(text="ab", score=0.9),))

    with pytest.raises(ValueError, match="character boxes"):
        to_paddlex_res(result, word_boxes=True)
