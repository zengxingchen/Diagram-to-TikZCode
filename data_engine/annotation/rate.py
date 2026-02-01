"""
Rate the visual quality of TikZ-rendered figures (1-5) using a VLM
(default: ``Qwen/Qwen2.5-VL-32B-Instruct`` via vLLM).

The model is prompted with a 5-point rubric and produces a structured
text reply (Description / Strengths / Weaknesses / Overall / Score).
Each input row appends one JSONL record::

    {"index": <row idx>, "path": <image path>, "raw": <model text>}
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from vlm_runner import VLMBackend, run_batched


SYSTEM_PROMPT = (
    "You are an expert in scientific visualization and academic figure "
    "evaluation. Your task is to analyze scientific/technical figures "
    "(mostly rendered from TikZ code in arXiv papers) and rate their "
    "quality. Evaluate only the image itself, not any underlying code."
)

USER_PROMPT = """Here is a figure. Please analyze it and give a score from 1 to 5 according to the following rules:

Scoring Rubric

1 (Very Poor)
The figure completely fails to convey useful information.
Elements are unreadable, heavily overlapping, misplaced, or missing.
Layout is broken, with clutter, or obstructed content.
May consist of only a few random numbers, lines, or fragments that do not form a meaningful figure.
Overall design shows no adherence to academic or professional figure standards.

2 (Poor)
The figure conveys only partial information, but major flaws dominate.
Typical issues: incorrect or misleading axes, missing or misaligned labels, text too small to read, missing legend, or severe clutter.
Information can be extracted with effort, but readability is poor.
Gives an impression of being unpolished and unprofessional.

3 (Fair)
The main intended content is present and can be understood, but significant problems remain.
Issues may include misleading scales, crowded layout, low visual clarity, poor color contrast, or omission of key components.
Functionally acceptable but far from ideal in readability and aesthetics.
Belongs to the "understandable but uncomfortable to read" category.

4 (Good)
The figure is largely correct, clear, and informative.
Minor issues exist: slightly cramped text, labels not optimally placed, or a design that is functional but simple.
Overall professional and trustworthy, though not at the highest level of polish.
Belongs to the "acceptable and satisfactory" category.

5 (Excellent)
The figure is clear, precise, and publication-ready.
Layout is well-structured, elements are properly aligned, and scales are accurate.
Text is legible, legends are complete, and color choices are effective and distinguishable.
No major flaws: the design is both aesthetically pleasing and highly effective at communication.
Represents a "ready-to-publish" high-quality academic figure.

Output Format (always follow this structure):
Description: Briefly describe what the figure shows (1 to 2 sentences).
Strengths: List 1 to 2 positive aspects.
Weaknesses: List 1 to 2 negative aspects.
Overall evaluation: Write 2 to 4 sentences explaining the overall quality and why you assigned this score (mention both positive and negative aspects).
Score: Final rating in the format Score: X
"""


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rate the quality of TikZ-rendered figures (1-5) via a VLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input parquet")
    parser.add_argument(
        "--output-jsonl", required=True,
        help="Output JSONL with one record per image",
    )
    parser.add_argument(
        "--image-col", default="image",
        help="Source column with image bytes / paths / data URIs",
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH,
        help="vLLM model path or HuggingFace id",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of rows to rate")
    parser.add_argument(
        "--normalize-workers", type=int, default=os.cpu_count() or 8,
        help="Threads used to spill image bytes to disk + verify",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--no-resume", action="store_true",
                        help="Overwrite existing output instead of resuming")
    args = parser.parse_args(argv)

    backend = VLMBackend(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
    )
    run_batched(
        parquet_path=args.input,
        save_jsonl=args.output_jsonl,
        backend=backend,
        inputs_builder=lambda b, path: b.build_llm_inputs(
            path, SYSTEM_PROMPT, USER_PROMPT,
        ),
        image_col=args.image_col,
        batch_size=args.batch_size,
        limit_first_n=args.limit,
        normalize_workers=args.normalize_workers,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
