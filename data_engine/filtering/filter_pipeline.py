#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_pipeline.py

Pipeline stages
---------------
1) filter: blacklist -> density threshold -> single pdflatex compile + log
           checks -> write keep.parquet / dump.parquet
2) round : read keep.parquet, round numeric literals to 4 significant figures,
           write rounded.parquet
3) sample: read rounded.parquet, bin by token count (HuggingFace tokenizer),
           write sampled.parquet
4) both  : filter -> round -> sample in one run

Examples
--------
# filter only
python filter_pipeline.py filter \\
  --input  data.parquet \\
  --out_keep keep.parquet \\
  --out_dump dump.parquet \\
  --workers 32 --chunksize 16 --timeout 15

# round only
python filter_pipeline.py round \\
  --input  keep.parquet \\
  --output rounded.parquet

# sample only (from rounded parquet)
python filter_pipeline.py sample \\
  --input  rounded.parquet \\
  --out_sample sampled.parquet \\
  --hf_tokenizer ./hf_tokenizer_cache \\
  --bin_strategy random --seed 42 \\
  --samples_per_bin 10000 \\
  --bins "0-200,200-400,400-600,600-800,800-1200,1200-2000,2000-4096"

# end-to-end (filter -> round -> sample)
python filter_pipeline.py both \\
  --input  data.parquet \\
  --out_keep keep.parquet --out_dump dump.parquet \\
  --out_rounded rounded.parquet --out_sample sampled.parquet \\
  --workers 16 --chunksize 8 --timeout 20 \\
  --hf_tokenizer ./hf_tokenizer_cache \\
  --bin_strategy random --seed 42 --samples_per_bin 10000

The first time `--hf_tokenizer` is used, the tokenizer is downloaded
from HuggingFace ("Qwen/Qwen2.5-VL-7B-Instruct" by default) and cached
under that directory; subsequent runs load offline.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from multiprocessing import Pool, cpu_count
from os import getpgid, killpg
from signal import SIGKILL
from subprocess import DEVNULL, Popen, TimeoutExpired
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

TQDM_KW = dict(
    dynamic_ncols=True, mininterval=0.2, position=0,
    leave=True, file=sys.stdout, ascii=True,
)


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "spiece.model", "vocab.txt",
)


def ensure_tokenizer_local(local_dir: str, model_id: str = DEFAULT_MODEL_ID):
    """Load (or download-then-load) a HuggingFace tokenizer.

    - If ``local_dir`` already has tokenizer files, load from there.
    - Otherwise download ``model_id`` and persist into ``local_dir``.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers not available: {exc}")

    has_files = os.path.isdir(local_dir) and any(
        os.path.exists(os.path.join(local_dir, f)) for f in _TOKENIZER_FILES
    )
    if not has_files:
        os.makedirs(local_dir, exist_ok=True)
        print(f"[tokenizer] Not found under {local_dir}. Downloading {model_id}...")
        AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        ).save_pretrained(local_dir)
        print(f"[tokenizer] Saved to {local_dir}")

    print(f"[tokenizer] Using local tokenizer from {local_dir}")
    return AutoTokenizer.from_pretrained(
        local_dir, local_files_only=True, trust_remote_code=True,
    )


# ---------------------------------------------------------------------------
# Filtering rules: blacklist + density + LaTeX log patterns
# ---------------------------------------------------------------------------

BLACKLIST_KEYWORDS = [
    "rnd", "pgfmathsetseed", "bricks", "random", "includegraphics",
    "tikzducks", "tikzpeople", "tikzsymbols", "AddToHook",
    "lipsum", "example-image", "newpage",
    "animate", "addtobeamertemplate", "frame", "overlay-beamer-styles",
    "blindtext", "minipage", "TikzBox", "tikzlings", "pgfornament",
    "fontawesome", "tikzposter", "figure", "makeatletter", "pig",
    "tikzstyle", "filecontents",
]
KW_LOWER = [k.lower() for k in BLACKLIST_KEYWORDS]

DEFAULT_LATEX_TIMEOUT = 20
DEFAULT_CHUNKSIZE = 256

PGFPLOTS_WARN_RE = re.compile(
    r"Package pgfplots Warning:.*?"
    r"(out of range|out of bounds|clipp|dropped|discarded|empty plot|"
    r"could not be transformed)",
    re.IGNORECASE,
)
TIKZ_WARN_RE = re.compile(
    r"Package tikz Warning:.*?"
    r"(outside|not allowed outside|dimension too large|overflow|"
    r"out of range|discarded)",
    re.IGNORECASE,
)
OVERFULL_WARN_RE = re.compile(r"Overfull", re.IGNORECASE)
ERROR_RES = [
    re.compile(r"! LaTeX Error:.*"),
    re.compile(r"! Package .* Error:.*"),
    re.compile(r"! Undefined control sequence.*"),
    re.compile(r"! Missing .*"),
    re.compile(r"! Emergency stop\."),
]

SERIES_POINT_LIMIT = 16   # max points per single addplot series
AXIS_TOTAL_LIMIT = 96     # max total points per axis

# Generic number pattern used by density & rounding regexes
NUM = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"

TIKZ_BLOCK_RE = re.compile(r"\\begin{tikzpicture}.*?\\end{tikzpicture}", re.DOTALL)
AXIS_BLOCK_RE = re.compile(r"\\begin{axis}.*?\\end{axis}", re.DOTALL)
COORD_PAIR_RE = re.compile(rf"\(\s*{NUM}\s*,\s*{NUM}(?:\s*,\s*{NUM})?\s*\)")
COORDINATES_BLOCK_CAP_RE = re.compile(
    r"((?:coordinates|plot\s+coordinates)\s*\{)(.*?)(\})",
    re.DOTALL | re.IGNORECASE,
)
# allow optional table options like [row sep=crcr]
TABLE_BLOCK_CAP_RE = re.compile(
    r"((?:\\addplot[\s\S]*?table(?:\[[^\]]*\])?\s*\{)|"
    r"(?:\\pgfplotstableread\s*\{))(.*?)(\})",
    re.DOTALL | re.IGNORECASE,
)
TWO_NUMS_LINE_RE = re.compile(
    rf"^\s*{NUM}\s+{NUM}\s*(?:\\\\)?\s*(?:%.*)?$", re.MULTILINE,
)
TIKZ_PLOT_COORD_CAP_RE = re.compile(
    r"((?:\\draw|\\path)[^{};]*?plot[^{};]*?coordinates\s*\{)(.*?)(\})",
    re.DOTALL | re.IGNORECASE,
)


def _count_points_series_text(series_text: str) -> int:
    pairs = COORD_PAIR_RE.findall(series_text)
    return len(pairs) if pairs else len(TWO_NUMS_LINE_RE.findall(series_text))


def is_too_dense(code: str) -> bool:
    if not code:
        return False
    for axis_block in AXIS_BLOCK_RE.findall(code):
        axis_total = 0
        for blk_re in (COORDINATES_BLOCK_CAP_RE, TABLE_BLOCK_CAP_RE):
            for m in blk_re.finditer(axis_block):
                cnt = _count_points_series_text(m.group(2))
                if cnt > SERIES_POINT_LIMIT:
                    return True
                axis_total += cnt
        if axis_total > AXIS_TOTAL_LIMIT:
            return True
    for m in TIKZ_PLOT_COORD_CAP_RE.finditer(code):
        if _count_points_series_text(m.group(2)) > SERIES_POINT_LIMIT:
            return True
    return False


# ---------------------------------------------------------------------------
# LaTeX compile (one-shot pdflatex + log scan)
# ---------------------------------------------------------------------------

def _latex_run(cmd: List[str], cwd: str, timeout: int) -> int:
    with Popen(cmd, cwd=cwd, stdout=DEVNULL, stderr=DEVNULL,
               start_new_session=True) as p:
        try:
            p.communicate(timeout=timeout)
            return p.returncode or 0
        except TimeoutExpired:
            rc = 124
        except Exception:
            rc = 1
        try:
            killpg(getpgid(p.pid), SIGKILL)
        finally:
            p.wait()
        return rc


def compile_and_check_log_pdflatex(code: str, timeout: int) -> Tuple[bool, bool]:
    """Run pdflatex once and parse main.log for warnings / errors.

    Returns ``(compile_ok, hit_bad_warn_or_error)``.
    """
    lines = code.split("\n")
    lines.insert(1, r"\thispagestyle{empty}\pagestyle{empty}")
    with TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "main.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        cmd = [
            "pdflatex", "-interaction=batchmode", "-halt-on-error",
            "-no-shell-escape", "-file-line-error", "main.tex",
        ]
        rc = _latex_run(cmd, cwd=tmpdir, timeout=timeout)
        log_path = os.path.join(tmpdir, "main.log")
        log_text = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="latin-1", errors="ignore") as lf:
                log_text = lf.read()

    compile_ok = (rc == 0)
    bad = (
        bool(PGFPLOTS_WARN_RE.search(log_text))
        or bool(TIKZ_WARN_RE.search(log_text))
        or bool(OVERFULL_WARN_RE.search(log_text))
        or any(rx.search(log_text) for rx in ERROR_RES)
        or (rc != 0)
    )
    return compile_ok, bad


# ---------------------------------------------------------------------------
# 4-significant-figures rounding
# ---------------------------------------------------------------------------

def _round_num_str_keep_plus(num_str: str) -> str:
    """Round one numeric literal to 4 significant figures, preserving sign
    prefix, leading-zero integers, scientific notation and the original
    integer-vs-decimal style."""
    lead_plus = num_str.startswith("+")
    # Respect original scientific notation: do not alter
    if "e" in num_str.lower():
        return num_str
    try:
        val = float(num_str)
    except Exception:
        return num_str

    body = num_str.lstrip("+-")
    # Preserve integers with leading zeros (e.g. 01, 002) exactly
    if body.isdigit() and len(body) > 1 and body.startswith("0"):
        return num_str

    has_decimal_point = "." in body
    decimals = len(body.split(".", 1)[1]) if has_decimal_point else 0
    leading_dot_style = has_decimal_point and body.startswith(".")

    # Value-wise round to 4 significant figures
    if val == 0:
        v_rounded = 0.0
    else:
        p = 4 - int(math.floor(math.log10(abs(val)))) - 1
        v_rounded = round(val, max(p, 0))

    # Reformat preserving original style
    if has_decimal_point:
        out = f"{v_rounded:.{decimals}f}"
        if leading_dot_style:
            if out.startswith("0."):
                out = out[1:]
            elif out.startswith("-0."):
                out = "-" + out[2:]
    else:
        if abs(v_rounded - int(v_rounded)) < 1e-12:
            out = str(int(v_rounded))
        else:
            out = str(v_rounded)

    if lead_plus and not out.startswith(("+", "-")):
        out = "+" + out
    return out


# Boundary-safe number pattern: never matches numbers preceded by '.'
_NUM_CORE = r"(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?"
_LEFT_B = r"(?:(?<=^)|(?<=[\s,;=\(\{\[\+\-]))"
_RIGHT_B = r"(?=$|[\s,;:\)\}\]\%])"
SAFE_NUM_RE = re.compile(rf"{_LEFT_B}(\+?-?{_NUM_CORE}){_RIGHT_B}")


def round_numbers_in_span(text: str) -> str:
    return SAFE_NUM_RE.sub(lambda m: _round_num_str_keep_plus(m.group(1)), text)


# definecolor's third {...}
DEF_COLOR_RGB_RE = re.compile(
    r"(\\definecolor\{[^}]+\}\{rgb\}\{)([^}]*)(\})", re.DOTALL,
)
# Bare (x,y[,z]) tuples, used outside coordinates/table blocks
PAREN_COORD_SPAN_RE = re.compile(
    r"(\()([^()]*(?:\([^()]*\)[^()]*)*)(\))"
)
# Strict (num, num[, num]) only
STRICT_COORD_TUPLE_RE = re.compile(
    rf"^\s*{NUM}\s*,\s*{NUM}(?:\s*,\s*{NUM})?\s*$"
)
# TikZ key=numeric-value options
TIKZ_KV_NUM_RE = re.compile(
    r"([A-Za-z@/_]+)\s*=\s*([+\-]?" + _NUM_CORE + r")(?=$|[\s,;\]\}])"
)
# Identifier-style keys whose values must never be rounded
_FORBID_NUMERIC_KV_KEYS = {
    "name", "id", "node", "label", "classname", "class", "style",
    "prefix", "suffix", "compat", "version",
}
# Only these KV keys are rounded; everything else is left untouched.
_ALLOW_NUMERIC_KV_KEYS = {
    "pos", "out", "in", "xshift", "yshift", "shift", "rotate", "scale",
    "xscale", "yscale", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax",
    "opacity", "line width", "inner sep", "outer sep",
}


def _sub_block(m):
    """Replacement for any capture group regex of shape (pre)(body)(suf)."""
    pre, body, suf = m.group(1), m.group(2), m.group(3)
    return f"{pre}{round_numbers_in_span(body)}{suf}"


def _sub_paren_coords(m):
    pre, body, suf = m.group(1), m.group(2), m.group(3)
    if STRICT_COORD_TUPLE_RE.match(body):
        return f"{pre}{round_numbers_in_span(body)}{suf}"
    return m.group(0)


def _sub_kv_nums(text: str) -> str:
    def kv_repl(m):
        key, val = m.group(1), m.group(2)
        key_l = key.lower()
        if "name" in key_l or key_l in _FORBID_NUMERIC_KV_KEYS:
            return m.group(0)
        if key_l not in _ALLOW_NUMERIC_KV_KEYS:
            return m.group(0)
        return f"{key}={_round_num_str_keep_plus(val)}"
    return TIKZ_KV_NUM_RE.sub(kv_repl, text)


def round_numbers_in_code(code: str, sigfigs: int = 4) -> str:
    """Round numeric literals in safe regions of TikZ / PGFPlots code.

    ``sigfigs`` is currently fixed to 4; the parameter is kept for API
    compatibility.
    """
    if not code:
        return code
    s = code
    s = DEF_COLOR_RGB_RE.sub(_sub_block, s)            # definecolor rgb
    s = COORDINATES_BLOCK_CAP_RE.sub(_sub_block, s)    # axis coordinates
    s = TABLE_BLOCK_CAP_RE.sub(_sub_block, s)          # table / pgfplotstableread
    s = PAREN_COORD_SPAN_RE.sub(_sub_paren_coords, s)  # bare (x, y[, z])
    s = _sub_kv_nums(s)                                # TikZ option=number
    return s


# ---------------------------------------------------------------------------
# Parallel filter stages
# ---------------------------------------------------------------------------

# A "predicate worker" takes ``(orig_idx, payload)`` and returns
# ``(orig_idx, keep_bool)``. ``payload`` is whatever was packed in the
# task tuple after the index.

def _blacklist_predicate(args) -> Tuple[int, bool]:
    orig_idx, code = args
    try:
        cl = (code or "").lower()
        return orig_idx, not any(k in cl for k in KW_LOWER)
    except Exception:
        return orig_idx, False


def _density_predicate(args) -> Tuple[int, bool]:
    orig_idx, code = args
    try:
        return orig_idx, not is_too_dense(code)
    except Exception:
        return orig_idx, False


def _compile_predicate(args) -> Tuple[int, bool]:
    orig_idx, code, timeout = args
    try:
        ok, bad = compile_and_check_log_pdflatex(code, timeout=timeout)
        return orig_idx, (ok and not bad)
    except Exception:
        return orig_idx, False


def _run_stage(
    predicate: Callable,
    tasks: Sequence,
    workers: int,
    chunksize: int,
    desc: str,
) -> Tuple[List[int], List[int]]:
    """Run ``predicate`` over ``tasks`` in parallel and split orig_ids
    into (keep, drop) lists, both sorted and deduped."""
    keep: List[int] = []
    drop: List[int] = []
    with Pool(processes=workers) as pool:
        for orig_idx, ok in tqdm(
            pool.imap_unordered(predicate, tasks, chunksize=chunksize),
            total=len(tasks), desc=desc, **TQDM_KW,
        ):
            (keep if ok else drop).append(orig_idx)
    return sorted(set(keep)), sorted(set(drop))


# ---------------------------------------------------------------------------
# Stage 1: filter
# ---------------------------------------------------------------------------

def run_filter(
    input_path: str, out_keep: str, out_dump: str,
    workers: int, chunksize: int, timeout: int,
) -> str:
    print(f"Loading: {input_path}")
    ds = load_dataset("parquet", data_files=input_path, split="train")
    if "orig_id" not in ds.column_names:
        ds = ds.add_column("orig_id", list(range(len(ds))))
    if "code" not in ds.column_names:
        raise ValueError("Input dataset missing 'code' column.")
    print(f"Total rows: {len(ds)}")

    # Columnar extract once; per-row dataset indexing is much slower.
    orig_ids = [int(x) for x in ds["orig_id"]]
    codes = ds["code"]

    # 1) blacklist
    keep_ids, drop_black = _run_stage(
        _blacklist_predicate,
        list(zip(orig_ids, codes)),
        workers, chunksize,
        desc=f"Blacklist filter (w={workers}, chunk={chunksize})",
    )
    print(f"After blacklist: keep={len(keep_ids)} | drop={len(drop_black)}")

    # 2) density (skipped automatically when keep_ids is empty)
    keep_ids, drop_dense = _run_stage(
        _density_predicate,
        [(oi, codes[oi]) for oi in keep_ids],
        workers, chunksize,
        desc=f"Density filter (SERIES>{SERIES_POINT_LIMIT} "
             f"or AXIS>{AXIS_TOTAL_LIMIT})",
    ) if keep_ids else ([], [])
    print(f"After density: keep={len(keep_ids)} | drop+={len(drop_dense)}")

    # 3) compile + log check
    keep_ids, drop_compile = _run_stage(
        _compile_predicate,
        [(oi, codes[oi], timeout) for oi in keep_ids],
        workers, chunksize,
        desc=f"Compile+log check (w={workers}, chunk={chunksize})",
    ) if keep_ids else ([], [])
    print(f"After compile+log: keep={len(keep_ids)} | drop+={len(drop_compile)}")

    # Write splits
    dump_set = sorted(set(drop_black) | set(drop_dense) | set(drop_compile))
    _ensure_parent_dir(out_keep)
    _ensure_parent_dir(out_dump)
    ds.select(keep_ids).to_parquet(out_keep)
    ds.select(dump_set).to_parquet(out_dump)
    print(f"Saved kept dataset  -> {out_keep} (rows={len(keep_ids)})")
    print(f"Saved dumped dataset -> {out_dump} (rows={len(dump_set)})")
    return out_keep


# ---------------------------------------------------------------------------
# Stage 2: round
# ---------------------------------------------------------------------------

def run_round(input_path: str, output_path: str) -> None:
    print(f"Loading: {input_path}")
    ds = load_dataset("parquet", data_files=input_path, split="train")
    if "code" not in ds.column_names:
        raise ValueError("Input dataset missing 'code' column.")

    def _round_row(ex):
        ex["code"] = round_numbers_in_code(ex["code"], sigfigs=4)
        return ex

    print("Rounding numbers to 4 significant figures...")
    ds_rounded = ds.map(_round_row, desc="Round numbers to 4 sig figs")
    _ensure_parent_dir(output_path)
    ds_rounded.to_parquet(output_path)
    print(f"Saved rounded dataset -> {output_path} (rows={len(ds_rounded)})")


# ---------------------------------------------------------------------------
# Stage 3: sample (HuggingFace tokenizer required)
# ---------------------------------------------------------------------------

DEFAULT_BINS: List[Tuple[int, int]] = [
    (0, 200), (200, 400), (400, 600), (600, 800),
    (800, 1200), (1200, 2000), (2000, 4096),
]


def run_sample(
    filtered_keep_path: str, out_sample: str, hf_tokenizer: str,
    seed: int, bins: Sequence[Tuple[int, int]],
    samples_per_bin: int, bin_strategy: str,
) -> None:
    if not hf_tokenizer:
        print("FATAL: --hf_tokenizer is required for the sample stage.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading filtered keep: {filtered_keep_path}")
    ds = load_dataset("parquet", data_files=filtered_keep_path, split="train")
    for col in ("orig_id", "code"):
        if col not in ds.column_names:
            raise ValueError(f"keep parquet must include '{col}' column.")
    print(f"Kept rows: {len(ds)}")

    # Columnar extract (much faster than indexing row-by-row).
    orig_ids = [int(x) for x in ds["orig_id"]]
    codes = ds["code"]
    id2pos = {oid: i for i, oid in enumerate(orig_ids)}

    try:
        hf_tok = ensure_tokenizer_local(
            local_dir=hf_tokenizer, model_id=DEFAULT_MODEL_ID,
        )
    except Exception as exc:
        print(f"FATAL: failed to prepare/load tokenizer locally: {exc}",
              file=sys.stderr)
        sys.exit(1)

    # Tokenize + bin by token count
    buckets = {f"{lo}-{hi}": [] for (lo, hi) in bins}
    for oid, code in tqdm(
        zip(orig_ids, codes), total=len(orig_ids),
        desc="Tokenize & bin (HF)", **TQDM_KW,
    ):
        try:
            t = len(hf_tok.encode(code or "", add_special_tokens=False))
        except Exception:
            t = 0
        for lo, hi in bins:
            if lo <= t < hi:
                buckets[f"{lo}-{hi}"].append(oid)
                break

    # Sample within each bin
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for lo, hi in bins:
        key = f"{lo}-{hi}"
        arr = sorted(set(buckets[key]))
        if not arr:
            print(f"Bin {key}: selected 0 / 0")
            continue
        if bin_strategy == "first":
            pick = arr[:samples_per_bin]
        else:
            k = min(samples_per_bin, len(arr))
            pick = rng.choice(arr, size=k, replace=False).tolist()
        selected.extend(pick)
        print(f"Bin {key}: selected {len(pick)} / {len(arr)}")

    pos_indices = sorted({id2pos[oid] for oid in selected if oid in id2pos})
    ds_sampled = ds.select(pos_indices)
    _ensure_parent_dir(out_sample)
    ds_sampled.to_parquet(out_sample)
    print(f"Saved sampled dataset -> {out_sample} (rows={len(ds_sampled)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_filter_args(p: argparse.ArgumentParser, *, in_keep_dump: bool) -> None:
    p.add_argument("--input", required=True, help="Input parquet path")
    if in_keep_dump:
        p.add_argument("--out_keep", required=True,
                       help="Output parquet for kept rows")
        p.add_argument("--out_dump", required=True,
                       help="Output parquet for dumped rows")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    p.add_argument("--timeout", type=int, default=DEFAULT_LATEX_TIMEOUT)


def _add_sample_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hf_tokenizer", type=str, required=True,
                   help="Path to a local HuggingFace tokenizer directory")
    p.add_argument("--bin_strategy", choices=["random", "first"], default="random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples_per_bin", type=int, default=10000)
    p.add_argument("--bins", type=str, default="",
                   help="Comma-separated ranges like '0-200,200-400,...' "
                        "(default: built-in 7-bin layout)")


def _parse_bins_arg(bins_arg: str) -> List[Tuple[int, int]]:
    if not bins_arg:
        return DEFAULT_BINS
    out: List[Tuple[int, int]] = []
    for seg in bins_arg.split(","):
        seg = seg.strip()
        if not seg:
            continue
        lo, hi = seg.split("-")
        out.append((int(lo), int(hi)))
    return out or DEFAULT_BINS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter (blacklist + density + pdflatex log) and "
                    "Sample (token-bin) a parquet of LaTeX/TikZ code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p_filter = sub.add_parser("filter", help="run filtering stage only")
    _add_filter_args(p_filter, in_keep_dump=True)

    p_round = sub.add_parser(
        "round", help="run rounding stage only (on filtered keep parquet)",
    )
    p_round.add_argument("--input", required=True,
                         help="Filtered keep parquet path")
    p_round.add_argument("--output", required=True,
                         help="Output for rounded parquet")

    p_sample = sub.add_parser(
        "sample", help="run sampling stage only (on filtered keep parquet)",
    )
    p_sample.add_argument("--input", required=True,
                          help="Filtered keep parquet path")
    p_sample.add_argument("--out_sample", required=True,
                          help="Output sampled parquet path")
    _add_sample_args(p_sample)

    p_both = sub.add_parser("both", help="run filter -> round -> sample")
    _add_filter_args(p_both, in_keep_dump=True)
    p_both.add_argument("--out_rounded", required=True,
                        help="Output parquet for kept rows (rounded)")
    p_both.add_argument("--out_sample", required=True,
                        help="Output sampled parquet path")
    _add_sample_args(p_both)

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.stage == "filter":
        run_filter(args.input, args.out_keep, args.out_dump,
                   workers=args.workers, chunksize=args.chunksize,
                   timeout=args.timeout)

    elif args.stage == "round":
        run_round(args.input, args.output)

    elif args.stage == "sample":
        run_sample(args.input, args.out_sample,
                   hf_tokenizer=args.hf_tokenizer,
                   seed=args.seed, bins=_parse_bins_arg(args.bins),
                   samples_per_bin=args.samples_per_bin,
                   bin_strategy=args.bin_strategy)

    elif args.stage == "both":
        print("--- Running Filter Stage ---")
        keep_path = run_filter(args.input, args.out_keep, args.out_dump,
                               workers=args.workers, chunksize=args.chunksize,
                               timeout=args.timeout)
        print("\n--- Running Round Stage ---")
        run_round(keep_path, args.out_rounded)
        print("\n--- Running Sample Stage ---")
        run_sample(args.out_rounded, args.out_sample,
                   hf_tokenizer=args.hf_tokenizer,
                   seed=args.seed, bins=_parse_bins_arg(args.bins),
                   samples_per_bin=args.samples_per_bin,
                   bin_strategy=args.bin_strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
