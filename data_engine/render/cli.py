"""
Render LaTeX / TikZ source code to PNG.

Two subcommands sharing the same multiprocess engine:

* ``dir``      — render every ``.tex`` in a directory to ``<name>.png``
* ``parquet``  — render a code column of a parquet file into a new
                 image column (PNG bytes, HF ``Image`` feature)

Usage::

    python cli.py dir     --input tex_dir --output png_dir --workers 8
    python cli.py parquet --input data.parquet --output out.parquet \\
                          --code-col new_code --image-col render_image
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from render import tex2png


# ---------------------------------------------------------------------------
# Shared worker
# ---------------------------------------------------------------------------

def _init_worker() -> None:
    # Avoid each process spawning a fan-out of BLAS threads.
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _compile_one(args: Tuple[Any, str, int, int]) -> Tuple[Any, Optional[bytes], Optional[str]]:
    """Compile a single piece of LaTeX code.

    Returns ``(key, png_bytes_or_none, error_or_none)`` so that the
    caller can correlate results with whatever ``key`` it submitted.
    """
    key, code, size, timeout = args
    try:
        png = tex2png(
            code, size=size, timeout=timeout,
            expand_to_square=True, verbose=False,
        )
        return key, png, None
    except Exception as exc:  # noqa: BLE001
        return key, None, str(exc)


def _run_pool(
    jobs: List[Tuple[Any, str, int, int]],
    n_workers: int,
    desc: str,
    ordered: bool = False,
) -> Iterable[Tuple[Any, Optional[bytes], Optional[str]]]:
    method = "imap" if ordered else "imap_unordered"
    with Pool(n_workers, initializer=_init_worker) as pool:
        mapper = getattr(pool, method)(_compile_one, jobs)
        for item in tqdm(mapper, total=len(jobs), desc=desc):
            yield item


def _resolve_workers(workers: int) -> int:
    return workers if workers > 0 else cpu_count()


# ---------------------------------------------------------------------------
# Subcommand: dir
# ---------------------------------------------------------------------------

def cmd_dir(args: argparse.Namespace) -> int:
    tex_files = sorted(glob(os.path.join(args.input, "*.tex")))
    if not tex_files:
        print(f"No .tex files found in {args.input}")
        return 1
    os.makedirs(args.output, exist_ok=True)

    n_workers = _resolve_workers(args.workers)
    print(f"Rendering {len(tex_files)} files with {n_workers} workers...")

    jobs: List[Tuple[Any, str, int, int]] = []
    for path in tex_files:
        with open(path, "r", encoding="utf-8") as f:
            jobs.append((path, f.read(), args.size, args.timeout))

    failures: List[Tuple[str, str]] = []
    for path, png, err in _run_pool(jobs, n_workers, desc="Rendering"):
        if png is None:
            failures.append((path, err or "unknown"))
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(args.output, f"{name}.png"), "wb") as f:
            f.write(png)

    n_ok = len(tex_files) - len(failures)
    print(f"Successfully rendered {n_ok}/{len(tex_files)} files -> {args.output}")
    if failures:
        print(f"\n{len(failures)} files failed:")
        for path, err in failures[:10]:
            print(f"  {path}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    return 0 if not failures else 2


# ---------------------------------------------------------------------------
# Subcommand: parquet
# ---------------------------------------------------------------------------

def cmd_parquet(args: argparse.Namespace) -> int:
    # Lazy-import datasets only when actually needed.
    from datasets import Dataset, load_dataset
    from datasets import Image as HFImage

    ds = load_dataset("parquet", data_files=args.input)["train"]
    if args.code_col not in ds.column_names:
        raise SystemExit(
            f"Column '{args.code_col}' not in parquet. "
            f"Available: {ds.column_names}"
        )

    n_workers = _resolve_workers(args.workers)
    print(f"Rendering {len(ds)} rows with {n_workers} workers...")

    jobs = [
        (i, ds[i][args.code_col], args.size, args.timeout)
        for i in range(len(ds))
    ]
    images: Dict[int, Optional[bytes]] = {}
    for idx, png, _err in _run_pool(
        jobs, n_workers,
        desc=f"Rendering {args.code_col} -> {args.image_col}",
    ):
        images[idx] = png

    n_ok = sum(1 for v in images.values() if v is not None)
    print(f"Successfully compiled {n_ok}/{len(ds)} rows")

    if args.success_json:
        with open(args.success_json, "w") as f:
            json.dump(
                sorted(i for i, v in images.items() if v is not None),
                f, indent=2,
            )
        print(f"Saved success indices -> {args.success_json}")

    columns = ds.column_names

    def _slice(indices: List[int], with_image: bool) -> Dataset:
        data = {col: [ds[i][col] for i in indices] for col in columns}
        features = ds.features.copy()
        if with_image:
            data[args.image_col] = [images[i] for i in indices]
            features[args.image_col] = HFImage()
        return Dataset.from_dict(data, features=features)

    if args.drop_failed:
        out_ds = _slice(
            [i for i in range(len(ds)) if images[i] is not None],
            with_image=True,
        )
    else:
        out_ds = _slice(list(range(len(ds))), with_image=True)

    out_ds.to_parquet(args.output)
    print(f"Saved {len(out_ds)} rows -> {args.output}")

    if args.failed_output:
        failed = [i for i in range(len(ds)) if images[i] is None]
        if failed:
            failed_ds = _slice(failed, with_image=False)
            failed_ds.to_parquet(args.failed_output)
            print(f"Saved {len(failed_ds)} failed rows -> {args.failed_output}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=384,
                        help="Rendered PNG size (square padded)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-compilation timeout (seconds)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (0 = cpu_count())")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render LaTeX / TikZ code to PNG (directory or parquet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dir = sub.add_parser(
        "dir", help="Render every .tex in a directory to .png",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_dir.add_argument("--input", required=True, help="Input directory")
    p_dir.add_argument("--output", required=True, help="Output directory")
    _add_common(p_dir)
    p_dir.set_defaults(func=cmd_dir)

    p_pq = sub.add_parser(
        "parquet", help="Render a code column in a parquet to an image column",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_pq.add_argument("--input", required=True, help="Input parquet")
    p_pq.add_argument("--output", required=True, help="Output parquet")
    p_pq.add_argument("--code-col", default="code",
                      help="Source column with LaTeX/TikZ code")
    p_pq.add_argument("--image-col", default="render_image",
                      help="Output column for the rendered PNG bytes")
    p_pq.add_argument("--failed-output",
                      help="If given, write rows that failed to compile here")
    p_pq.add_argument("--success-json",
                      help="If given, write a JSON list of successful row indices")
    p_pq.add_argument("--drop-failed", action="store_true",
                      help="Drop failed rows from --output")
    _add_common(p_pq)
    p_pq.set_defaults(func=cmd_parquet)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
