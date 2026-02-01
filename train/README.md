# DaVinci RL training (reward + workers)

This folder contains the **reward function** and the **subprocess workers**
used to fine-tune Qwen2.5-VL with [GRPO](https://arxiv.org/abs/2402.03300)
(via [EasyR1](https://github.com/hiyouga/EasyR1)) on the TikZ30K dataset.

The reward measures *how visually faithful* a generated TikZ
program is to its ground-truth diagram, by aggregating four complementary
signals on the rendered output, matching the hybrid reward described in
the paper (Section 3.3):

```
text in <answer>``` ```
       │
       ▼
   pdflatex compile  ──►  PNG  +  PDF
       │                       │
       ├── MSE      (image)    ├── PDF text-box layout IoU       (Stage 4, R_text)
       └── DreamSim (image)    └── PDF geometric-element match   (Stage 5, R_geom)
       ─────────── R_img ───
```

Compile success (R_pass) is handled implicitly: if compilation fails,
every other component is assigned its minimum possible value, so the
overall reward collapses to its lowest. The four components are
combined without special weighting (each carries weight 0.5 in the code;
this is a uniform scale and is absorbed by GRPO's advantage
normalization). See `reward_function/reward.py` near the bottom of
`compute_score()` to tweak.


## Layout

```
train/
├── README.md
├── requirements.txt
├── train.sh                          example EasyR1 launch script
├── prompt_template/
│   └── tikz_nothink.jinja            user prompt (no chain-of-thought)
├── reward_function/
│   └── reward.py                     compute_score(reward_inputs) entry point
├── dreamsim_worker.py                ──┐
├── pdf_text_worker.py                  │  invoked by reward.py via
└── geometry_sim_batch_worker.py      ──┘  `python <worker>.py in.json out.json`
```

`reward.py` exposes a single function `compute_score(reward_inputs, ...)`
with the signature expected by EasyR1's
`worker.reward.reward_type=batch`.


## Setup

### 1. Trainer environment (the one that runs `train.sh`)

```bash
pip install -r requirements.txt
# Plus the trainer itself; follow EasyR1's own README for its full env:
#   https://github.com/hiyouga/EasyR1
```

You also need `pdflatex` + `latexmk` + `poppler` available on the system
(used by `_run_tikz_compilation` to render predictions).

### 2. DreamSim environment (separate conda env)

DreamSim pins different versions of `torch` / `torchvision` than the
trainer, so it has to live in its own conda env. Create it once:

```bash
conda create -n dreamsim python=3.10
conda activate dreamsim
pip install dreamsim torch torchvision pillow
```

Then, before launching training, point the reward at it:

```bash
export DREAMSIM_ENV=$(conda info --base)/envs/dreamsim
```

(Optionally set `DREAMSIM_CACHE_DIR` to control where the model weights are
downloaded; defaults to `~/.cache/dreamsim`.)


## Training data format

`reward.py` expects every record in `data.train_files` to carry, in addition
to whatever your trainer already needs (`image`, `code`):

| Field                     | Meaning                                  |
|---------------------------|------------------------------------------|
| `multi_modal_data.images` | the ground-truth PNG (any size)          |
| `ground_truth_pdf_bytes`  | the raw PDF bytes that produced the PNG  |

The PDF bytes are needed by the layout-IoU and geometry-similarity workers
to extract text boxes and vector elements without re-rendering.


## Run

Edit the placeholder paths near the top of `train.sh`:

```bash
REPO_ROOT      = /path/to/your/repo containing EasyR1/
MODEL_PATH     = /path/to/sft_checkpoint            # e.g. cold-start SFT
TRAIN_PARQUET  = /path/to/train.parquet
VAL_PARQUET    = /path/to/test.parquet
DREAMSIM_ENV   = /path/to/conda/envs/dreamsim
```

Then:

```bash
bash train/train.sh
```


## Reward score breakdown

`compute_score(reward_inputs)` returns one dict per sample, e.g.:

```python
{
    "overall":                       0.62,
    "format":                        1.0,
    "compilation":                   1,
    "reconstruction":                0.71,
    "reconstruction_non_edge_aware": 0.71,
    "dreamsim":                      0.83,
    "pdf_text":                      0.40,
    "geometry_sim":                  0.46,
}
```

`overall` is the value EasyR1 uses as the GRPO advantage signal; the rest
are logged for diagnostics.
