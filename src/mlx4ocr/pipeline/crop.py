"""Perspective cropping for detected text regions."""

from __future__ import annotations

import cv2
import numpy as np

from mlx4ocr.types import CharBox, TextDetection, TextRecognition


def sorted_box_indices(boxes: np.ndarray) -> np.ndarray:
    """Return indices sorting quadrilateral boxes top-to-bottom, left-to-right.

    Args:
        boxes: Box corners with shape ``[N, 4, 2]``.

    Returns:
        Index array that reorders ``boxes`` into reading order.
    """
    if boxes.ndim != 3 or boxes.shape[1:] != (4, 2):
        raise ValueError(f"expected boxes [N, 4, 2], got {boxes.shape}")

    num_boxes = boxes.shape[0]
    indices = sorted(range(num_boxes), key=lambda index: (boxes[index, 0, 1], boxes[index, 0, 0]))
    ordered = list(indices)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(boxes[ordered[j + 1], 0, 1] - boxes[ordered[j], 0, 1]) < 10 and (
                boxes[ordered[j + 1], 0, 0] < boxes[ordered[j], 0, 0]
            ):
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            else:
                break
    return np.asarray(ordered, dtype=np.int64)


def sorted_detections(detections: tuple[TextDetection, ...]) -> tuple[TextDetection, ...]:
    """Sort detections in reading order.

    Args:
        detections: Unordered DB detections.

    Returns:
        Detections sorted top-to-bottom, then left-to-right.
    """
    if not detections:
        return ()
    boxes = np.asarray(
        [[[x, y] for x, y in detection.box.points] for detection in detections],
        dtype=np.float32,
    )
    order = sorted_box_indices(boxes)
    return tuple(detections[int(index)] for index in order)


def _rect_size(points: np.ndarray) -> tuple[int, int]:
    """Return the integer ``(width, height)`` of the rectified crop for ``points``."""
    if points.shape != (4, 2):
        raise ValueError(f"expected four corner points, got shape {points.shape}")

    crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    return crop_width, crop_height


def _is_rotated(points: np.ndarray) -> bool:
    """Whether ``get_rotate_crop_image`` turns the crop by 90 degrees."""
    crop_width, crop_height = _rect_size(points)
    return crop_height / float(crop_width) >= 1.5


def _minarea_ordered_points(points: np.ndarray) -> np.ndarray:
    """Order the min-area rectangle of ``points`` as TL, TR, BR, BL."""
    bounding_box = cv2.minAreaRect(points.astype(np.int32))
    corner_points = sorted(cv2.boxPoints(bounding_box), key=lambda item: item[0])

    if corner_points[1][1] > corner_points[0][1]:
        index_a, index_d = 0, 1
    else:
        index_a, index_d = 1, 0
    if corner_points[3][1] > corner_points[2][1]:
        index_b, index_c = 2, 3
    else:
        index_b, index_c = 3, 2

    return np.array(
        [
            corner_points[index_a],
            corner_points[index_b],
            corner_points[index_c],
            corner_points[index_d],
        ],
        dtype=np.float32,
    )


def get_rotate_crop_image(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop and rectify a quadrilateral text region via perspective transform.

    Args:
        image: Source BGR image ``[H, W, 3]``.
        points: Four corner points with shape ``[4, 2]``.

    Returns:
        Rectified crop in BGR layout.
    """
    crop_width, crop_height = _rect_size(points)
    pts_std = np.float32(
        [
            [0, 0],
            [crop_width, 0],
            [crop_width, crop_height],
            [0, crop_height],
        ]
    )
    transform = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    crop = cv2.warpPerspective(
        image,
        transform,
        (crop_width, crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if _is_rotated(points):
        crop = np.rot90(crop)
    return crop


def get_minarea_rect_crop(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop a text region using the minimum-area bounding rectangle.

    Args:
        image: Source BGR image ``[H, W, 3]``.
        points: Four corner points with shape ``[4, 2]``.

    Returns:
        Rectified crop in BGR layout.
    """
    return get_rotate_crop_image(image, _minarea_ordered_points(points))


def crop_text_regions(
    image: np.ndarray,
    detections: tuple[TextDetection, ...],
    *,
    box_type: str = "quad",
) -> tuple[np.ndarray, ...]:
    """Crop detected text regions from a source image.

    Args:
        image: Source BGR image ``[H, W, 3]``.
        detections: Sorted detections whose boxes define crop regions.
        box_type: ``quad`` for perspective crop or ``poly`` for min-area crop.

    Returns:
        One BGR crop per detection.
    """
    if box_type not in {"quad", "poly"}:
        raise ValueError(f"box_type must be 'quad' or 'poly', got {box_type!r}")

    crops: list[np.ndarray] = []
    for detection in detections:
        points = np.array(detection.box.points, dtype=np.float32)
        if box_type == "quad":
            crops.append(get_rotate_crop_image(image, points))
        else:
            crops.append(get_minarea_rect_crop(image, points))
    return tuple(crops)


def char_boxes_in_image(
    detection: TextDetection,
    recognition: TextRecognition,
    *,
    box_type: str = "quad",
) -> tuple[CharBox, ...]:
    """Map CTC character cells from crop space back to image coordinates.

    Each cell is one CTC time step wide, centred on the character's peak, and
    spans the full height of the detected region. Rotated crops (see
    ``get_rotate_crop_image``) read along the box's left-to-right edge pair
    instead of its top-to-bottom one.

    Args:
        detection: Detection the recognition was cropped from.
        recognition: Recognition decoded with ``valid_ratios``.
        box_type: Same crop mode that produced the recognition crop.

    Returns:
        One ``(x_min, y_min, x_max, y_max)`` box per ``recognition.chars`` entry.
    """
    points = np.array(detection.box.points, dtype=np.float32)
    if box_type == "poly":
        points = _minarea_ordered_points(points)
    top_left, top_right, bottom_right, bottom_left = points
    if _is_rotated(points):
        start_a, end_a, start_b, end_b = top_left, bottom_left, top_right, bottom_right
    else:
        start_a, end_a, start_b, end_b = top_left, top_right, bottom_left, bottom_right

    half = recognition.char_width / 2.0
    boxes: list[CharBox] = []
    for center in recognition.char_centers:
        lo, hi = max(0.0, center - half), min(1.0, center + half)
        corners = np.array(
            [
                start_a + lo * (end_a - start_a),
                start_a + hi * (end_a - start_a),
                start_b + hi * (end_b - start_b),
                start_b + lo * (end_b - start_b),
            ]
        )
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        boxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
    return tuple(boxes)
