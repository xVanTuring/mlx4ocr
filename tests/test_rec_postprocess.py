"""Recognition CTC decode parity tests."""

from __future__ import annotations

import numpy as np

from mlx4ocr.postprocess.ctc import ctc_decode, load_character_dict
from tests.conftest import DICT_ROOT, GOLDEN_ROOT, load_golden_npy
from tests.reference.postprocess.ctc_decode import CTCLabelDecode


def test_ctc_decode_matches_reference_on_golden_softmax(variant: str) -> None:
    """MLX CTC decode matches vendored reference on golden softmax outputs."""
    softmax = load_golden_npy(GOLDEN_ROOT / variant / "rec" / "softmax.npy")
    dict_path = DICT_ROOT / ("ppocrv6_tiny_dict.txt" if variant == "tiny" else "ppocrv6_dict.txt")
    reference = CTCLabelDecode(character_dict_path=dict_path, use_space_char=True)(softmax)
    decoded = ctc_decode(softmax, load_character_dict(dict_path, use_space_char=True))
    assert len(decoded) == len(reference)
    for actual, (text, score) in zip(decoded, reference, strict=True):
        assert actual.text == text
        np.testing.assert_allclose(actual.score, score, rtol=1e-5, atol=1e-5)


def test_ctc_decode_blank_sequence() -> None:
    """An all-blank prediction decodes to empty text with zero confidence."""
    preds = np.zeros((1, 4, 8), dtype=np.float32)
    preds[:, :, 0] = 1.0
    dict_path = DICT_ROOT / "ppocrv6_dict.txt"
    decoded = ctc_decode(preds, load_character_dict(dict_path, use_space_char=True))
    assert decoded[0].text == ""
    assert decoded[0].score == 0.0


def test_ctc_decode_char_positions_skip_padding() -> None:
    """Peak time steps become line fractions that ignore right padding."""
    characters = ("blank", "a", "b")
    # 10 time steps, the first 5 belong to the crop: a . a-repeat b .
    sequence = [1, 1, 0, 2, 0, 0, 0, 0, 0, 0]
    preds = np.zeros((1, 10, 3), dtype=np.float32)
    preds[0, np.arange(10), sequence] = 1.0

    decoded = ctc_decode(preds, characters, valid_ratios=[0.5])[0]

    assert decoded.text == "ab"
    assert decoded.chars == ("a", "b")
    # First "a" peak is step 0, "b" is step 3; the crop spans 5 steps.
    np.testing.assert_allclose(decoded.char_centers, (0.5 / 5, 3.5 / 5))
    np.testing.assert_allclose(decoded.char_width, 1 / 5)


def test_ctc_decode_without_valid_ratios_has_no_positions() -> None:
    preds = np.zeros((1, 4, 3), dtype=np.float32)
    preds[0, :, 1] = 1.0
    decoded = ctc_decode(preds, ("blank", "a", "b"))[0]
    assert decoded.chars == ()
    assert decoded.char_centers == ()
