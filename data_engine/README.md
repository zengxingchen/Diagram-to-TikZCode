# data_engine

A four-stage data pipeline for building a clean, normalized, annotated
dataset of `(image, code)` TikZ pairs.

```
filtering/    blacklist + density + pdflatex log filter
              4-significant-figure rounding
              token-binned random sampling

normalize/    LLM rewrites the TikZ code:
                reorder.py        natural plotting order
                add_comments.py   inline annotations

render/       compile code to PNG:
                cli.py dir       dir of .tex     -> dir of .png
                cli.py parquet   parquet column  -> image column

annotation/   VLM labels each rendered image:
                classify.py   11-way figure category
                rate.py       1–5 visual-quality score
```

For preamble cleanup (pruning redundant `\usepackage` and
`\usetikzlibrary` declarations) see the standalone module
[`tikz_import_optimization/`](../tikz_import_optimization).

---

## Setup

```bash
pip install -r requirements.txt
```

`vllm` requires a CUDA GPU and is only needed for `annotation/`.

The `render/` stage additionally needs a system LaTeX toolchain:

- macOS: `brew install --cask mactex` (or `basictex`)
- Ubuntu: `apt install texlive-full poppler-utils ghostscript`

---

## Stages

Each script accepts `--help` for the full option list. Use any
filenames you like for inputs and outputs.

### `filtering/filter_pipeline.py`

Four sub-stages selectable via positional command: `filter`, `round`,
`sample`, or `both` (end-to-end).

```bash
python filtering/filter_pipeline.py both \
    --input  IN.parquet \
    --out_keep    keep.parquet \
    --out_dump    dump.parquet \
    --out_rounded rounded.parquet \
    --out_sample  sampled.parquet \
    --hf_tokenizer <tokenizer_dir_or_hf_id>
```

The first time `--hf_tokenizer` runs, the tokenizer is downloaded from
HuggingFace (default `Qwen/Qwen2.5-VL-7B-Instruct`) and cached locally;
subsequent runs load offline.

### `normalize/reorder.py` and `normalize/add_comments.py`

OpenAI-compatible LLM rewrites. Default backend is Aliyun DashScope;
pass `--base-url` and `--model-id` to use any OpenAI-style endpoint.

```bash
export OPENAI_API_KEY=sk-...

python normalize/reorder.py      --input IN.parquet --output OUT.parquet
python normalize/add_comments.py --input IN.parquet --output OUT.parquet
```

### `render/cli.py`

```bash
python render/cli.py dir     --input TEX_DIR  --output PNG_DIR
python render/cli.py parquet --input IN.parquet --output OUT.parquet \
                             --code-col CODE_COL --image-col IMAGE_COL
```

### `annotation/classify.py` and `annotation/rate.py`

vLLM batched inference (default `Qwen/Qwen2.5-VL-32B-Instruct`).
Reads images from a parquet, writes JSONL.

```bash
python annotation/classify.py --parquet IN.parquet --output category.jsonl
python annotation/rate.py     --parquet IN.parquet --output quality.jsonl
```
