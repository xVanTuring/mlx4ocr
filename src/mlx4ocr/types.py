"""Frozen domain records for mlx4ocr inference outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned quadrilateral in image coordinates.

    Points are ordered clockwise starting from the top-left corner.
    """

    points: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

    @property
    def x_min(self) -> float:
        """Minimum x coordinate across all corners."""
        return min(point[0] for point in self.points)

    @property
    def y_min(self) -> float:
        """Minimum y coordinate across all corners."""
        return min(point[1] for point in self.points)

    @property
    def x_max(self) -> float:
        """Maximum x coordinate across all corners."""
        return max(point[0] for point in self.points)

    @property
    def y_max(self) -> float:
        """Maximum y coordinate across all corners."""
        return max(point[1] for point in self.points)


@dataclass(frozen=True)
class TextDetection:
    """Single detected text region."""

    box: BoundingBox
    score: float


@dataclass(frozen=True)
class TextRecognition:
    """Recognized text for one cropped region."""

    text: str
    score: float
    # Per-character CTC peak positions as fractions of the crop's reading axis
    # (0 = start of the line, 1 = end). ``chars`` joins back to ``text``.
    # Empty when the decoder was not asked for positions.
    chars: tuple[str, ...] = field(default=(), compare=False)
    char_centers: tuple[float, ...] = field(default=(), compare=False)
    # Width of one CTC time step, as a fraction of the reading axis.
    char_width: float = field(default=0.0, compare=False)


CharBox = tuple[float, float, float, float]
"""Axis-aligned ``(x_min, y_min, x_max, y_max)`` box in image coordinates."""


@dataclass(frozen=True)
class OCRTextBlock:
    """Recognized text block in an OCR result."""

    text: str
    box: BoundingBox | None = None
    detection_score: float | None = None
    recognition_score: float | None = None
    # Per-character cells in image coordinates; ``chars`` joins back to ``text``.
    chars: tuple[str, ...] = field(default=(), compare=False)
    char_boxes: tuple[CharBox, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class OCRResult:
    """End-to-end OCR output for one image."""

    blocks: tuple[OCRTextBlock, ...]
    text: str
    engine: str
    model: str | None = None
    prompt: str | None = None

    @classmethod
    def from_ppocrv6(
        cls,
        detections: tuple[TextDetection, ...],
        recognitions: tuple[TextRecognition, ...],
        *,
        model: str | None = None,
        char_boxes: tuple[tuple[CharBox, ...], ...] | None = None,
    ) -> OCRResult:
        """Build a PP-OCRv6 result from aligned detection and recognition tuples.

        Args:
            detections: Filtered text detections in reading order.
            recognitions: Filtered text recognitions aligned with ``detections``.
            model: Optional model variant or identifier.
            char_boxes: Optional per-block character cells aligned with
                ``detections``; each entry matches ``recognition.chars``.

        Returns:
            Generalized block-based OCR result.
        """
        if char_boxes is not None and len(char_boxes) != len(detections):
            raise ValueError("char_boxes must align with detections")
        blocks = tuple(
            OCRTextBlock(
                text=recognition.text,
                box=detection.box,
                detection_score=detection.score,
                recognition_score=recognition.score,
                chars=recognition.chars if char_boxes is not None else (),
                char_boxes=char_boxes[index] if char_boxes is not None else (),
            )
            for index, (detection, recognition) in enumerate(
                zip(detections, recognitions, strict=True)
            )
        )
        return cls(
            blocks=blocks,
            text="\n".join(block.text for block in blocks),
            engine="ppocrv6",
            model=model,
        )
