"""
Classify TikZ-rendered figures into one of a fixed set of categories
using a VLM (default: ``Qwen/Qwen2.5-VL-32B-Instruct`` via vLLM).

The model is prompted to emit one-line JSON of the form::

    {"label": <one of the categories>, "score": <float in [0,1]>, "brief": <=20 words}

Each input row produces one JSONL line in the output. Optionally
post-aggregates per-category counts to a CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

from vlm_runner import VLMBackend, run_batched


# ===========================================================================
# Category catalog + system prompt
# ===========================================================================

CATEGORY_DESCRIPTORS: Dict[str, List[str]] = {
    "flowcharts": [
        "a flowchart diagram with rectangular process boxes, decision diamonds, and arrows",
        "a process flow diagram with start and end terminators and directional arrows",
        "a swimlane flowchart with lanes and labeled steps",
        "a state machine diagram with states and transitions",
        "a sequence diagram with lifelines and messages",
    ],
    "charts": [
        "a scientific chart or plot with axes, ticks, and a legend",
        "a line chart of time series with gridlines",
        "a bar chart with categories on the x-axis",
        "a scatter plot with points and labels",
        "a heatmap figure with a colorbar",
    ],
    "circuits": [
        "an electrical circuit schematic diagram with resistors capacitors and ground",
        "an electronic circuit with op-amp symbol and signal flow",
        "a block diagram of an electrical system with arrows between blocks",
        "a digital logic circuit diagram with AND OR NOT gates",
        "a circuit diagram using standardized electronic symbols",
    ],
    "structures": [
        "a hierarchical structure diagram or tree with parent and child nodes",
        "an organizational chart with boxes and connecting lines",
        "a commutative diagram in category theory with arrows between objects",
        "a UML class diagram with relationships",
        "a tree diagram with multiple levels",
    ],
    "math_geometry": [
        "a geometric construction with angle arcs perpendicular marks and labels",
        "a math figure with coordinate axes labeled points and vectors",
        "a geometry sketch with circles lines and intersections",
        "a function illustration with annotations on a coordinate plane",
        "a trigonometry diagram with angles and right triangles",
    ],
    "physics": [
        "a physics free body diagram with force vectors and labels",
        "an optics ray diagram with lenses mirrors and light rays",
        "an electromagnetism diagram with field lines or circuits",
        "a kinematics diagram showing velocity acceleration vectors",
        "a physics textbook schematic with arrows and physical quantities",
    ],
    "chemistry": [
        "a chemistry structure diagram with benzene rings and bonds",
        "a chemical reaction mechanism with curved arrows",
        "a skeletal formula of organic molecules",
        "a reaction pathway diagram with reagents and products",
        "a chemistry textbook figure with molecular structures and labels",
    ],
    "networks": [
        "a computer network topology diagram with servers clients and routers",
        "a systems architecture diagram of microservices with arrows and data flow",
        "a distributed system diagram with nodes queues and databases",
        "a cloud architecture diagram with components and connections",
        "a data pipeline diagram with stages",
    ],
    "graphs_theory": [
        "a graph theory diagram with nodes and edges",
        "a shortest path illustration on a graph",
        "a network graph drawing with labeled vertices",
        "a directed acyclic graph diagram",
        "a planar graph with highlighted paths",
    ],
    "layouts_ui": [
        "a user interface wireframe with boxes and layout grids",
        "a page layout or typesetting mockup with columns",
        "a matrix of nodes layout diagram",
        "a dashboard wireframe with panes",
        "a layout grid with aligned elements",
    ],
    "mixed": [
        "an ambiguous technical diagram combining multiple types",
        "a mixed diagram that is hard to categorize",
        "a general technical figure with various elements",
        "an abstract schematic with multiple styles",
        "a complex diagram with mixed semantics",
    ],
}


def build_system_prompt(descriptors: Dict[str, List[str]]) -> str:
    lines = [
        "You are a specialist for classifying TikZ-rendered technical "
        "figures. Given ONE image, pick exactly ONE best-matching "
        "category from the list below.",
        "Categories with guidance:",
    ]
    for cat, descs in descriptors.items():
        lines.append(f"- {cat}:")
        for d in descs:
            lines.append(f"  • {d}")
    lines.append(
        "\nDecision rules:\n"
        "1) Choose the single most specific category that fits the overall figure.\n"
        "2) If it clearly mixes multiple types or is ambiguous, use 'mixed'.\n"
        "3) Prefer domain-specific categories (e.g., circuits, chemistry) "
           "over generic structures if symbols match.\n"
        "4) For standard scientific charts (axes/ticks/legends), use 'charts'.\n"
        "5) For flow-oriented boxes/diamonds/arrows, use 'flowcharts'.\n"
        "6) For geometry/math axes, vectors, constructions, use 'math_geometry'.\n"
        "7) Output MUST be a single-line compact JSON: "
        "{\"label\": <one of the categories>, \"score\": float in [0,1], \"brief\": \"<=20 words\"}.\n"
        "8) Do not include any additional text besides the JSON line.\n"
        "9) For 'mixed', briefly explain why in 'brief'."
    )
    return "\n".join(lines)


SYSTEM_PROMPT = build_system_prompt(CATEGORY_DESCRIPTORS)
USER_PROMPT = (
    "Classify now. Reply ONLY with one-line JSON: "
    "{\"label\": <category>, \"score\": <0~1>, \"brief\": \"<=20 words\"}"
)


# ===========================================================================
# Optional aggregation
# ===========================================================================

def _parse_json_loose(raw: str):
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        # Try truncating the right brace to recover from trailing junk.
        right = candidate.rfind("}")
        while right > 0:
            try:
                return json.loads(candidate[: right + 1])
            except Exception:
                right = candidate.rfind("}", 0, right)
    return None


def aggregate_stats(jsonl_path: str, csv_path: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    total = 0
    parsed_ok = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                rec = {}

            label = None
            parsed = rec.get("parsed")
            if isinstance(parsed, dict) and "label" in parsed:
                label = str(parsed["label"]).strip()
                parsed_ok += 1
            else:
                loose = _parse_json_loose(rec.get("raw", ""))
                if isinstance(loose, dict) and "label" in loose:
                    label = str(loose["label"]).strip()
            if not label:
                label = "mixed"
            counts[label] = counts.get(label, 0) + 1

    print("\n=== Category Count Summary ===")
    for k in sorted(counts, key=lambda x: (-counts[x], x)):
        print(f"  {k:20s} : {counts[k]}")
    print(f"  TOTAL(lines) : {total}")
    print(f"  PARSED  ok   : {parsed_ok}")
    print(f"  FALLBACK     : {total - parsed_ok}")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,count\n")
        for k in sorted(counts, key=lambda x: (-counts[x], x)):
            f.write(f"{k},{counts[k]}\n")
    print(f"Saved stats CSV -> {csv_path}")
    return counts


# ===========================================================================
# CLI
# ===========================================================================

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify TikZ-rendered figures into categories via a VLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input parquet")
    parser.add_argument(
        "--output-jsonl", required=True,
        help="Output JSONL with one record per image",
    )
    parser.add_argument(
        "--stats-csv",
        help="Optional CSV: per-category aggregated counts",
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
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of rows to classify")
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
        resume=not args.no_resume,
    )

    if args.stats_csv:
        aggregate_stats(args.output_jsonl, args.stats_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
