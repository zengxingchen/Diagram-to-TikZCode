#!/bin/bash
# ---------------------------------------------------------------------------
# DaVinci RL training entry point.
#
# Reward = MSE + DSIM (image fidelity) + PDF text-layout IoU + PDF
# geometry similarity. See `reward_function/reward.py` for the breakdown.
#
# Required external tools / data (please fill in for your machine):
#   - REPO_ROOT          repository root containing EasyR1/
#   - MODEL_PATH         HF-format VLM checkpoint (e.g. cold-start SFT model)
#   - TRAIN_PARQUET      train split (must include a `pdf_bytes` column)
#   - VAL_PARQUET        eval  split
#   - DREAMSIM_ENV       conda env prefix where DreamSim is installed
#                        (DreamSim pins different torch versions, so it must
#                        run in its own env)
# ---------------------------------------------------------------------------

set -x
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export SWANLAB_MODE=${SWANLAB_MODE:-offline}

# --- Paths (please edit) ----------------------------------------------------
REPO_ROOT=${REPO_ROOT:-/path/to/DiagramRL}
MODEL_PATH=${MODEL_PATH:-/path/to/sft_checkpoint}
TRAIN_PARQUET=${TRAIN_PARQUET:-data/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-data/test.parquet}

# DreamSim runs in its own conda env (see reward_function/reward.py).
export DREAMSIM_ENV=${DREAMSIM_ENV:-/path/to/anaconda3/envs/dreamsim}

# Optional: resume from a previous checkpoint
# LOAD_CHECKPOINT_PATH=/path/to/checkpoints/global_step_300

EXPERIMENT_NAME=${EXPERIMENT_NAME:-davinci_geometry}

cd "$REPO_ROOT"

python3 -m EasyR1.verl.trainer.main \
    config=EasyR1/examples/config.yaml \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${VAL_PARQUET} \
    data.image_key=image \
    data.answer_key=code \
    data.rollout_batch_size=256 \
    data.format_prompt=train/prompt_template/tikz_nothink.jinja \
    data.max_response_length=6144 \
    worker.rollout.max_num_batched_tokens=9000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=train/reward_function/reward.py:compute_score \
    worker.rollout.n=10 \
    trainer.total_epochs=7 \
    trainer.project_name=DaVinci \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=true \
    trainer.val_generations_to_log=3 \
    ${LOAD_CHECKPOINT_PATH:+trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH}}
