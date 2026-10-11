"""End-to-end PP-OCRv6 detection and recognition pipeline."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx4ocr.hub.download import HubArtifacts, download_model
from mlx4ocr.hub.rec_weight_patch import RecognitionWeightSource
from mlx4ocr.hub.registry import ModelVariant
from mlx4ocr.models.det.model import DetectionModel
from mlx4ocr.models.rec.model import RecognitionModel
from mlx4ocr.output import OCRTiming
from mlx4ocr.pipeline.config import PipelineConfig, pipeline_config_from_artifacts
from mlx4ocr.pipeline.crop import char_boxes_in_image, crop_text_regions, sorted_detections
from mlx4ocr.pipeline.memory import MemoryPolicy, PipelineMemoryRuntime
from mlx4ocr.postprocess.ctc import ctc_decode
from mlx4ocr.postprocess.db import db_postprocess
from mlx4ocr.preprocess.det import (
    det_preprocess,
    nhwc_prob_to_nchw,
    normalize_det_image_mlx,
    resize_det_image,
)
from mlx4ocr.preprocess.rec import rec_preprocess
from mlx4ocr.types import CharBox, OCRResult, TextDetection, TextRecognition

logger = logging.getLogger(__name__)


def recognize_crops(
    recognizer: Callable[[mx.array], mx.array],
    crops: tuple[np.ndarray, ...],
    *,
    rec_image_shape: tuple[int, int, int],
    characters: tuple[str, ...],
    rec_batch_num: int,
    with_positions: bool = False,
) -> tuple[TextRecognition, ...]:
    """Run batched recognition on cropped text regions.

    Args:
        recognizer: Loaded recognition model or compiled recognition callable.
        crops: BGR crops in detection order.
        rec_image_shape: Base ``(C, H, W)`` shape from inference configs.
        characters: CTC character table with blank at index 0.
        rec_batch_num: Maximum crops per forward pass.
        with_positions: Also record each character's position along its crop.

    Returns:
        Recognition results in the same order as ``crops``.
    """
    if not crops:
        return ()

    width_ratios = np.asarray([crop.shape[1] / float(crop.shape[0]) for crop in crops])
    sort_indices = np.argsort(width_ratios)
    results: list[TextRecognition | None] = [None] * len(crops)

    for batch_start in range(0, len(crops), rec_batch_num):
        batch_end = min(len(crops), batch_start + rec_batch_num)
        batch_indices = sort_indices[batch_start:batch_end]
        max_wh_ratio = rec_image_shape[2] / float(rec_image_shape[1])
        for index in batch_indices:
            crop = crops[int(index)]
            max_wh_ratio = max(max_wh_ratio, crop.shape[1] / float(crop.shape[0]))

        # Crops are resized to the model height and right-padded to the batch's widest
        # crop; the share of time steps that belongs to the crop itself is this ratio.
        padded_w = int(rec_image_shape[1] * max_wh_ratio)
        valid_ratios = []
        for index in batch_indices:
            crop_h, crop_w = crops[int(index)].shape[:2]
            resized_w = min(math.ceil(rec_image_shape[1] * crop_w / crop_h), padded_w)
            valid_ratios.append(resized_w / padded_w)

        batch_tensors: list[mx.array] = []
        for index in batch_indices:
            preprocessed = rec_preprocess(
                crops[int(index)],
                max_wh_ratio=max_wh_ratio,
                rec_image_shape=rec_image_shape,
            )
            batch_tensors.append(preprocessed.image)

        batch_input = mx.concatenate(batch_tensors, axis=0)
        logits = recognizer(batch_input)
        mx.eval(logits)
        softmax = np.asarray(logits, dtype=np.float32)
        decoded = ctc_decode(
            softmax,
            characters,
            valid_ratios=valid_ratios if with_positions else None,
        )
        for offset, recognition in enumerate(decoded):
            original_index = int(batch_indices[offset])
            results[original_index] = recognition

    if any(result is None for result in results):
        raise RuntimeError("recognition produced incomplete batch results")
    return tuple(result for result in results if result is not None)


@dataclass(frozen=True)
class PipelineResult:
    """OCR output with stage timings."""

    result: OCRResult
    timing: OCRTiming


@dataclass(frozen=True)
class PP_OCRv6:
    """PP-OCRv6 detection and recognition pipeline."""

    variant: ModelVariant
    detector: DetectionModel
    recognizer: RecognitionModel
    config: PipelineConfig
    memory: PipelineMemoryRuntime
    detector_forward: Callable[[mx.array], mx.array] | None = None
    detector_preprocess_forward: Callable[[mx.array], mx.array] | None = None
    recognizer_forward: Callable[[mx.array], mx.array] | None = None

    @classmethod
    def from_artifacts(
        cls,
        variant: ModelVariant,
        det_artifacts: HubArtifacts,
        rec_artifacts: HubArtifacts,
        *,
        drop_score: float = 0.5,
        rec_batch_num: int = 6,
        det_box_type: str = "quad",
        rec_weight_source: RecognitionWeightSource = "auto",
        memory_policy: MemoryPolicy | None = None,
        compile_models: bool = True,
    ) -> PP_OCRv6:
        """Construct a pipeline from downloaded Hub artifacts.

        Args:
            variant: Model size tier.
            det_artifacts: Local detection Hub files.
            rec_artifacts: Local recognition Hub files.
            drop_score: Minimum recognition score to keep a detection.
            rec_batch_num: Maximum recognition batch size.
            det_box_type: ``quad`` or ``poly`` crop mode.
            rec_weight_source: Recognition weight loading mode.
            memory_policy: MLX allocator policy for inference and teardown.
            compile_models: Compile model forward passes with ``mx.compile``.

        Returns:
            Initialized pipeline with loaded weights.
        """
        config = pipeline_config_from_artifacts(
            det_artifacts,
            rec_artifacts,
            drop_score=drop_score,
            rec_batch_num=rec_batch_num,
            det_box_type=det_box_type,
        )
        memory = PipelineMemoryRuntime(memory_policy or MemoryPolicy())
        memory.apply_init_limits()
        detector = DetectionModel.from_artifacts(det_artifacts)
        recognizer = RecognitionModel.from_artifacts(
            rec_artifacts,
            weight_source=rec_weight_source,
        )
        detector_preprocess_forward: Callable[[mx.array], mx.array] | None = None
        if compile_models:

            def detector_with_preprocess(resized_image: mx.array) -> mx.array:
                return detector(normalize_det_image_mlx(resized_image))

            detector_preprocess_forward = mx.compile(detector_with_preprocess)
        return cls(
            variant=variant,
            detector=detector,
            recognizer=recognizer,
            config=config,
            memory=memory,
            detector_forward=mx.compile(detector) if compile_models else None,
            detector_preprocess_forward=detector_preprocess_forward,
            recognizer_forward=mx.compile(recognizer) if compile_models else None,
        )

    @classmethod
    def from_hub(
        cls,
        variant: ModelVariant,
        *,
        cache_dir: Path | None = None,
        det_variant: ModelVariant | None = None,
        rec_variant: ModelVariant | None = None,
        drop_score: float = 0.5,
        rec_batch_num: int = 6,
        det_box_type: str = "quad",
        rec_weight_source: RecognitionWeightSource = "auto",
        memory_policy: MemoryPolicy | None = None,
        compile_models: bool = True,
    ) -> PP_OCRv6:
        """Download Hub weights and construct a PP-OCRv6 pipeline.

        Args:
            variant: Model size tier.
            cache_dir: Optional Hugging Face cache directory.
            det_variant: Optional detection model tier. Defaults to ``variant``.
            rec_variant: Optional recognition model tier. Defaults to ``variant``.
            drop_score: Minimum recognition score to keep a detection.
            rec_batch_num: Maximum recognition batch size.
            det_box_type: ``quad`` or ``poly`` crop mode.
            rec_weight_source: Recognition weight loading mode. ``auto`` uses
                raw Hub safetensors for ``tiny`` and Paddle-pretrained head
                patches for ``small``/``medium``.
            memory_policy: MLX allocator policy for inference and teardown.
            compile_models: Compile model forward passes with ``mx.compile``.

        Returns:
            Initialized pipeline with loaded weights.
        """
        resolved_det_variant = det_variant or variant
        resolved_rec_variant = rec_variant or variant
        det_artifacts = download_model(resolved_det_variant, "det", cache_dir=cache_dir)
        rec_artifacts = download_model(resolved_rec_variant, "rec", cache_dir=cache_dir)
        return cls.from_artifacts(
            variant,
            det_artifacts,
            rec_artifacts,
            drop_score=drop_score,
            rec_batch_num=rec_batch_num,
            det_box_type=det_box_type,
            rec_weight_source=rec_weight_source,
            memory_policy=memory_policy,
            compile_models=compile_models,
        )

    def close(self) -> None:
        """Release MLX allocator cache held by this pipeline."""
        self.memory.release()

    def __call__(self, image: np.ndarray) -> OCRResult:
        """Run detection and recognition on a BGR image."""
        return self.predict(image).result

    def predict(self, image: np.ndarray, *, char_boxes: bool = False) -> PipelineResult:
        """Run detection and recognition and return timings.

        Args:
            image: Source image in BGR uint8 layout ``[H, W, 3]``.
            char_boxes: Also fill ``OCRTextBlock.chars`` / ``char_boxes`` with each
                character's cell in image coordinates.

        Returns:
            OCR output plus per-stage elapsed seconds.

        Raises:
            ValueError: If ``image`` is not a 3-channel BGR array.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected BGR image [H, W, 3], got shape {image.shape}")

        total_start = time.perf_counter()

        if self.detector_preprocess_forward is None:
            preprocessed = det_preprocess(
                image,
                limit_side_len=self.config.det_limit_side_len,
                limit_type=self.config.det_limit_type,
            )
            shape = preprocessed.shape
            det_input = preprocessed.image
        else:
            resized, shape = resize_det_image(
                image,
                limit_side_len=self.config.det_limit_side_len,
                limit_type=self.config.det_limit_type,
            )
            det_input = mx.array(resized)
        det_start = time.perf_counter()
        if self.detector_preprocess_forward is not None:
            prob_tensor = self.detector_preprocess_forward(det_input)
        else:
            detector = self.detector_forward or self.detector
            prob_tensor = detector(det_input)
        mx.eval(prob_tensor)
        prob_map = nhwc_prob_to_nchw(prob_tensor)
        det_s = time.perf_counter() - det_start
        self.memory.maybe_clear_after_det()

        detections = db_postprocess(
            prob_map,
            shape,
            thresh=float(self.config.det_postprocess_params["thresh"]),
            box_thresh=float(self.config.det_postprocess_params["box_thresh"]),
            max_candidates=int(self.config.det_postprocess_params["max_candidates"]),
            unclip_ratio=float(self.config.det_postprocess_params["unclip_ratio"]),
            score_mode=str(self.config.det_postprocess_params.get("score_mode", "fast")),
        )
        ordered = sorted_detections(detections)
        if not ordered:
            logger.info("No text regions detected")
            total_s = time.perf_counter() - total_start
            self.memory.on_predict_end()
            return PipelineResult(
                result=OCRResult.from_ppocrv6((), (), model=self.variant),
                timing=OCRTiming(det_s=det_s, rec_s=0.0, total_s=total_s),
            )

        crops = crop_text_regions(
            image,
            ordered,
            box_type=self.config.det_box_type,
        )
        rec_start = time.perf_counter()
        recognitions = recognize_crops(
            self.recognizer_forward or self.recognizer,
            crops,
            rec_image_shape=self.config.rec_image_shape,
            characters=self.config.characters,
            rec_batch_num=self.config.rec_batch_num,
            with_positions=char_boxes,
        )
        rec_s = time.perf_counter() - rec_start

        filtered_detections: list[TextDetection] = []
        filtered_recognitions: list[TextRecognition] = []
        filtered_char_boxes: list[tuple[CharBox, ...]] = []
        for detection, recognition in zip(ordered, recognitions, strict=True):
            if recognition.score < self.config.drop_score:
                continue
            filtered_detections.append(detection)
            filtered_recognitions.append(recognition)
            if char_boxes:
                filtered_char_boxes.append(
                    char_boxes_in_image(detection, recognition, box_type=self.config.det_box_type)
                )

        logger.info(
            "Detected %d regions, kept %d after drop_score=%.2f",
            len(ordered),
            len(filtered_detections),
            self.config.drop_score,
        )
        total_s = time.perf_counter() - total_start
        self.memory.on_predict_end()
        return PipelineResult(
            result=OCRResult.from_ppocrv6(
                tuple(filtered_detections),
                tuple(filtered_recognitions),
                model=self.variant,
                char_boxes=tuple(filtered_char_boxes) if char_boxes else None,
            ),
            timing=OCRTiming(det_s=det_s, rec_s=rec_s, total_s=total_s),
        )
