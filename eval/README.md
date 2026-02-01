# eval

Two-step evaluation for image-to-TikZ models: **run the MLLM on a test
parquet, then score the predictions** with a battery of text and image
metrics.

We evaluate on the **test split** of
[`nllg/datikz-v3`](https://huggingface.co/datasets/nllg/datikz-v3/blob/main/data/test-00000-of-00001.parquet)
(DaTi*k*Zv3, 542 examples), the same benchmark used by DeTi*k*Zify.

```
eval/
├── qwenvl_infer.py     vLLM batched inference (image → TikZ JSON)
├── run_eval.py         score predictions with text + image metrics
└── metrics/            10 individual metric implementations
```

## Setup

```bash
pip install -r requirements.txt
```

`qwenvl_infer.py` and most reconstruction metrics need a CUDA GPU. The
prediction PNGs (`--pred-dir`) and reference PNGs (`--ref-dir`) must be
rendered separately — see [`../data_engine/render`](../data_engine/render).

## Stages

### 1. Inference

```bash
python qwenvl_infer.py \
    --model-dir <hf_model_id_or_local_dir> \
    --parquet   IN.parquet \
    --save-path OUT.json \
    --tensor-parallel-size 4
```

The input parquet must contain an `image` column and a `code` column
(used as ground-truth in the output JSON).

### 2. Evaluation

```bash
python run_eval.py \
    --json       OUT.json \
    --pred-dir   PRED_PNG_DIR \
    --ref-dir    REF_PNG_DIR \
    --output-dir EVAL_DIR \
    --compute-cbleu --compute-ted \
    --compute-dsim --compute-siglip --compute-ssim --compute-psnr
```

`--compute-*` flags are opt-in; each adds one metric:

| Flag | Metric |
|------|--------|
| `--compute-cbleu` | CrystalBLEU (code n-gram overlap) |
| `--compute-ted` | TeX edit distance |
| `--compute-dsim` | DreamSim perceptual similarity |
| `--compute-siglip` | SigLIP cosine similarity |
| `--compute-kid` | KID over SigLIP features |
| `--compute-mse` | edge-based pixel MSE (Canny) |
| `--compute-mse-nocanny` | plain pixel MSE |
| `--compute-ssim` | SSIM |
| `--compute-lpips` | LPIPS perceptual distance |
| `--compute-psnr` | PSNR |

Outputs:

- `EVAL_DIR/metrics.csv` — per-image scores
- A printed summary of the aggregates.

## Acknowledgement

Most of the metric implementations under [`metrics/`](./metrics) are
adapted from [**DeTikZify**](https://github.com/potamides/DeTikZify).
If you use this evaluation code, please cite their paper:

```bibtex
@inproceedings{belouadi2024detikzify,
    title     = {{DeTikZify}: Synthesizing Graphics Programs for
                 Scientific Figures and Sketches with {TikZ}},
    author    = {Jonas Belouadi and Simone Paolo Ponzetto and Steffen Eger},
    booktitle = {The Thirty-eighth Annual Conference on Neural
                 Information Processing Systems},
    year      = {2024},
    url       = {https://openreview.net/forum?id=bcVLFQCOjc}
}
```
