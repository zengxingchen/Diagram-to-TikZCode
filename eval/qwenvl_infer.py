"""
Run a Qwen2.5-VL (or compatible MLLM) on a parquet of `(image, code)`
pairs and emit image-to-TikZ-code predictions as a JSON file.

The input parquet must contain an ``image`` column (PIL / bytes / URL)
and a ``code`` column (ground-truth TikZ code for reference).

Output JSON is a list of
``{index, predicted, ground_truth}`` records, one per row.

Example
-------
    python qwenvl_infer.py \
        --model-dir  Qwen/Qwen2.5-VL-7B-Instruct \
        --parquet    test.parquet \
        --save-path  results/eval.json \
        --tensor-parallel-size 4
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


DEFAULT_PROMPT = (
    "This is a picture of a scientific figure. Generate LaTeX code that "
    "draws this scientific figure using TikZ. Ensure that the LaTeX code "
    "is self-contained and does not require any packages except "
    "TikZ-related imports. Don't forget to include \\usepackage{tikz}! "
    "Return your result in a latex code block."
)


def get_image(sample: Dict[str, Any]) -> Image.Image:
    """Decode an image cell that may be a URL, bytes, dict, or PIL image."""
    img = sample["image"]
    if isinstance(img, str) and img.startswith("http"):
        return Image.open(requests.get(img, stream=True).raw).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return img


def unwrap_latex_code_block(text: str) -> str:
    """Strip the ``` / ```latex ... ``` fences from a model reply."""
    text = text.strip()
    if text.startswith("```latex"):
        text = text[len("```latex"):].lstrip("\n\r")
    elif text.startswith("```"):
        text = text[len("```"):].lstrip("\n\r")
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


def main(args: argparse.Namespace) -> int:
    llm = LLM(
        model=args.model_dir,
        limit_mm_per_prompt={"image": 10, "video": 10},
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_utilization,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=[],
    )

    dataset = pd.read_parquet(args.parquet)
    total_len = len(dataset)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    print(f"Loaded {total_len} rows from {args.parquet}")

    results = []
    pending_inputs = []
    pending_meta = []  # (index, ground_truth)

    def flush_batch() -> None:
        if not pending_inputs:
            return
        try:
            outputs = llm.generate(pending_inputs, sampling_params=sampling_params)
            for out, (ex_idx, gt_code) in zip(outputs, pending_meta):
                pred_code = unwrap_latex_code_block(out.outputs[0].text)
                results.append({
                    "index": ex_idx,
                    "predicted": pred_code,
                    "ground_truth": gt_code,
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[batch up to {len(results)}/{total_len}] batch error: {exc}")
        finally:
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[{len(results)}/{total_len}] processed")
            pending_inputs.clear()
            pending_meta.clear()

    for idx, (_, sample) in tqdm(enumerate(dataset.iterrows()), total=total_len):
        try:
            if isinstance(sample["image"], np.ndarray):
                sample["image"] = sample["image"][0]
            image = get_image(sample)
            gt_code = sample["code"]

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True,
            )

            mm_data: Dict[str, Any] = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            pending_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            })
            pending_meta.append((idx, gt_code))

            if len(pending_inputs) >= args.batch_size:
                flush_batch()
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx + 1}/{total_len}] error: {exc}")
            continue

    flush_batch()
    print(f"Saved {len(results)} predictions -> {args.save_path}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Qwen2.5-VL (or compatible MLLM) on a parquet of "
                    "image-code pairs and emit TikZ predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True,
                   help="HuggingFace model id or local checkpoint directory.")
    p.add_argument("--parquet", required=True,
                   help="Input parquet with 'image' and 'code' columns.")
    p.add_argument("--save-path", required=True,
                   help="Output JSON file with predictions.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="User prompt sent alongside each image.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="Number of GPUs for tensor parallelism. Must divide "
                        "the model's attention head count.")
    p.add_argument("--pipeline-parallel-size", type=int, default=1,
                   help="Number of GPUs for pipeline parallelism. "
                        "TP * PP must be <= visible GPUs.")
    p.add_argument("--gpu-mem-utilization", type=float, default=0.9,
                   help="vLLM GPU memory utilization ratio in (0, 1].")
    # Sampling
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.001)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-tokens", type=int, default=14096)
    p.add_argument("--seed", type=int, default=1)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # Sanity checks
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 1
    required = args.tensor_parallel_size * args.pipeline_parallel_size
    if required > visible:
        raise SystemExit(
            f"tensor_parallel_size ({args.tensor_parallel_size}) * "
            f"pipeline_parallel_size ({args.pipeline_parallel_size}) = "
            f"{required} exceeds visible GPUs ({visible})."
        )

    sys.exit(main(args))
