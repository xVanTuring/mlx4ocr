# mlx4ocr

[![Lint](https://github.com/shuuul/mlx4ocr/actions/workflows/lint.yml/badge.svg)](https://github.com/shuuul/mlx4ocr/actions/workflows/lint.yml)
[![PyPI](https://img.shields.io/pypi/v/mlx4ocr.svg)](https://pypi.org/project/mlx4ocr/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Apple MLX](https://img.shields.io/badge/Apple-MLX-black.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/116943)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Apple Silicon OCR powered by [MLX](https://github.com/ml-explore/mlx),
[PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6), and an
optional VLM backend for GLM-OCR and PaddleOCR-VL.

`mlx4ocr` reimplements PP-OCRv6 detection and recognition for local macOS
inference. It downloads official Hugging Face `safetensors` weights on demand
and runs the OCR pipeline without a PaddlePaddle runtime. For generated-text OCR,
it can also run GLM-OCR and PaddleOCR-VL through the optional `mlx-vlm` extra.

> [!NOTE]
> This project is pre-alpha. APIs and output details may change while the MLX
> port is being completed and validated.

## Features

- PP-OCRv6 text detection and recognition on Apple Silicon with MLX.
- Official `tiny`, `small`, and `medium` Hugging Face model variants.
- Image, PDF, and non-recursive directory inputs from the CLI.
- Plain text, Markdown, and PaddleOCR-style JSON output.
- Optional GLM-OCR and PaddleOCR-VL generated-text backends through `mlx-vlm`.
- Optional saved output layout compatible with document OCR workflows.
- Optional MCP server and installable agent skill for compatible coding agents.

## Requirements

- macOS on Apple Silicon.
- Python 3.12 or newer.
- [`uv`](https://docs.astral.sh/uv/) for local development and CLI execution.
- Internet access on first run to download model weights from Hugging Face.
- For optional VLM OCR, enough disk space for the selected VLM checkpoint. The
  GLM-OCR `mlx-community/GLM-OCR-bf16` main `model.safetensors` file is about
  2.2 GB; PaddleOCR-VL size is not yet benchmarked in this project.

## Installation

Install the CLI from PyPI with `uv tool`:

```bash
uv tool install mlx4ocr
```

Or install directly from GitHub:

```bash
uv tool install git+https://github.com/shuuul/mlx4ocr.git
```

Or run the CLI without installing it permanently:

```bash
uvx --from mlx4ocr mlx4ocr --help
```

For development from a checkout:

```bash
git clone https://github.com/shuuul/mlx4ocr.git
cd mlx4ocr
uv sync --group dev
```

Optional VLM OCR support through `mlx-vlm` is available as an extra:

```bash
uv sync --extra vlm
uv tool install 'mlx4ocr[vlm]'
```

## Quick start

Run OCR on an image and print Markdown to stdout:

```bash
mlx4ocr --path input.png --format markdown
```

From a development checkout, you can run the bundled examples with `uv run`:

```bash
uv run mlx4ocr --path examples/images/img_10.jpg --format markdown
```

Use `uvx` when running directly from GitHub without installation:

```bash
uvx --from mlx4ocr \
  mlx4ocr --path input.png --format markdown
```

Python API:

```python
import cv2

from mlx4ocr import PP_OCRv6

image = cv2.imread("examples/images/img_10.jpg")
ocr = PP_OCRv6.from_hub("medium")

try:
    result = ocr.predict(image)
    print(result.result.text)
    for block in result.result.blocks:
        print(block.text, block.box, block.detection_score, block.recognition_score)
    print(result.timing)
finally:
    ocr.close()
```

The Python `OCRResult` is block-based. `result.result.text` contains recognized
lines joined with newlines, and `result.result.blocks` contains `OCRTextBlock`
items with optional geometry and detection/recognition scores.

Per-character boxes are opt-in. `predict(image, char_boxes=True)` fills each
block's `chars` and `char_boxes` (`(x_min, y_min, x_max, y_max)` in image
pixels, one CTC time step wide, centred on the character's peak), and
`to_paddlex_res(result, word_boxes=True)` emits them as PaddleX
`text_word` / `text_word_boxes`. Centres are reliable; widths are not.

Optional VLM OCR API, using GLM-OCR or PaddleOCR-VL through `mlx-vlm`:

```python
from mlx4ocr import VLMOCR

# GLM-OCR is the default VLM preset.
ocr = VLMOCR.from_hub()

try:
    result = ocr.predict_path("examples/images/img_10.jpg")
    print(result.text)
finally:
    ocr.close()
```

Use PaddleOCR-VL by selecting the preset engine:

```python
from mlx4ocr import VLMOCR

ocr = VLMOCR.from_hub(engine="paddleocr-vl", task="chart")

try:
    result = ocr.predict_path("chart.png")
    print(result.text)
finally:
    ocr.close()
```

This API requires installing the `vlm` extra. It currently returns the generated
text as one `OCRTextBlock` without geometry, detection scores, or recognition
scores.

## CLI usage

The CLI accepts image files, PDF files, or a non-recursive directory of supported
inputs:

```bash
mlx4ocr --path examples/images --format json --output ocr-output
```

Supported output formats:

- `txt` — recognized text only.
- `markdown` — recognized text as Markdown, preserving PDF page headings.
- `json` — PaddleOCR/PaddleX-compatible PP-OCRv6 `res` fields with PDF
  `page_index` metadata. This format requires OCR blocks with geometry and
  detection/recognition scores.

PP-OCRv6 is the default CLI engine. Optional VLM CLI inference is available
through `mlx-vlm` after installing the `vlm` extra. Supported VLM presets are
GLM-OCR and PaddleOCR-VL:

```bash
uv sync --extra vlm
mlx4ocr --path input.png --engine glm-ocr --format markdown
mlx4ocr --path input.pdf --engine glm-ocr --vlm-task table --max-tokens 1024
mlx4ocr --path chart.png --engine paddleocr-vl --vlm-task chart --format markdown
```

Use `--vlm-model` to select a different compatible Hugging Face model. GLM-OCR
tasks are `text`, `formula`, `table`, and `schema`. PaddleOCR-VL tasks are
`text`, `formula`, `table`, `chart`, and `schema`. The `schema` task requires a
custom `--prompt` for both VLM engines. JSON output for VLM engines uses a
generalized `result` object with generated text and blocks, not the PaddleX
`res` geometry fields.

### Engine comparison and VLM resource notes

`mlx4ocr` has three local MLX OCR engine presets:

- `ppocrv6` — default detector/recognizer pipeline. It returns text blocks with
  geometry and detection/recognition scores.
- `glm-ocr` — optional VLM generated-text pipeline. It can produce more natural
  page text, but it does not return text boxes or confidence scores.
- `paddleocr-vl` — optional PaddleOCR-VL generated-text pipeline through
  `mlx-vlm`, including chart recognition prompts. It also does not return text
  boxes or confidence scores.

VLM engines are much heavier than the default PP-OCRv6 path. On first use,
`mlx-vlm` downloads the selected model. The GLM-OCR preset downloads
`mlx-community/GLM-OCR-bf16`; the main safetensors file is about 2.2 GB.
PaddleOCR-VL uses `PaddlePaddle/PaddleOCR-VL`; the local Hugging Face cache was
about 1.8 GB in the smoke test below. If a download stalls through Hugging
Face/Xet, retry with:

```bash
HF_HUB_DISABLE_XET=1 uv run --extra vlm mlx4ocr \
  --path examples/ppocrv6.pdf --engine glm-ocr --format markdown --start 0 --end 0
```

Smoke-test timings on the development machine for `examples/ppocrv6.pdf`, with
models already cached:

| Engine | Input | Options | Wall time | Max RSS |
|--------|-------|---------|-----------|---------|
| `ppocrv6` | page 1 | `--start 0 --end 0` | ~8.6 s | ~1.17 GB |
| `glm-ocr` | page 1 | `--start 0 --end 0 --max-tokens 128` | ~11 s | ~2.5 GB |
| `paddleocr-vl` | page 1 | `--start 0 --end 0 --max-tokens 128` | ~13.4 s | ~2.49 GB |
| `ppocrv6` | full 10-page PDF | default tokens N/A | ~1 min 21 s | ~1.30 GB |
| `glm-ocr` | full 10-page PDF | `--max-tokens 256` | ~3 min 37 s | ~3.1 GB |
| `paddleocr-vl` | full 10-page PDF | `--max-tokens 256` | ~5 min 21 s | ~2.48 GB |

These numbers are indicative rather than a guarantee. Actual time and memory
depend on the Mac, MLX version, image/PDF resolution, prompt, and `--max-tokens`.
Increase `--max-tokens` for long VLM pages to reduce truncation; expect
processing time to increase with the generated output length.

PDF page ranges use 0-based inclusive page indexes:

```bash
mlx4ocr --path docs/report.pdf --format markdown --start 0 --end 2
```

When `--output` is omitted, results are printed to stdout. When `--output` is
provided, files are written with this layout:

- `<output>/<stem>/ocr/<stem>.txt` — plain text output.
- `<output>/<stem>/ocr/<stem>.md` — Markdown output.
- `<output>/<stem>/ocr/<stem>.json` — JSON output.
- `<output>/<stem>/ocr/<stem>_origin.pdf` — original PDF copy for PDF inputs.

Useful options:

```bash
mlx4ocr --help
mlx4ocr --path input.png --variant tiny --format txt
mlx4ocr --path input.pdf --rec-weight-source auto --no-compile
```

## MCP server

Optional MCP support is available as an extra:

```bash
uv sync --extra mcp
uv run mlx4ocr-mcp
```

The MCP server exposes an `ocr_markdown` tool that reads a local image path and
returns Markdown OCR output.

## Agent skill

This repository includes an agent skill for compatible coding agents. Install it
with `npx skills`:

```bash
npx skills add shuuul/mlx4ocr
```

After installation, compatible agents can use the skill to run `mlx4ocr`
directly from GitHub with `uvx` or `uv tool` on macOS.

## Model variants

| Variant | Detection Hub repo | Recognition Hub repo |
|---------|-------------------|----------------------|
| `tiny` | `PaddlePaddle/PP-OCRv6_tiny_det_safetensors` | `PaddlePaddle/PP-OCRv6_tiny_rec_safetensors` |
| `small` | `PaddlePaddle/PP-OCRv6_small_det_safetensors` | `PaddlePaddle/PP-OCRv6_small_rec_safetensors` |
| `medium` | `PaddlePaddle/PP-OCRv6_medium_det_safetensors` | `PaddlePaddle/PP-OCRv6_medium_rec_safetensors` |

Use `det_variant` or `rec_variant` when detection and recognition should use
different tiers. For example, MinerU `ch_server` is closest to small detection
with medium recognition:

```python
from mlx4ocr import PP_OCRv6

ocr = PP_OCRv6.from_hub("medium", det_variant="small")
```

To download model artifacts without constructing the OCR pipeline:

```python
from mlx4ocr import download_model

artifacts = download_model("medium", "det")
print(artifacts.config_data["model_type"])
```

### Recognition weights for `small` and `medium`

The Hugging Face `small` and `medium` recognition safetensors currently ship
swapped `head.encoder.conv_block` tensors. By default,
`PP_OCRv6.from_hub(..., rec_weight_source="auto")` and
`load_recognition_model()` patch the affected tensors from official Paddle
pretrained checkpoints. The checkpoints are downloaded once to
`.cache/paddle_pretrained/`; PaddlePaddle is not required.

Use `rec_weight_source="hub"` only when you explicitly want the raw Hugging Face
safetensors.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run prek run --all-files
```

See [AGENTS.md](AGENTS.md) for architecture notes and coding conventions.

## License

MIT License. See [LICENSE](LICENSE).
