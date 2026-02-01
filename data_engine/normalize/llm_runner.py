"""
Async LLM rewriter for a column in a parquet file.

Given:
- an input parquet,
- the column with the source code to rewrite,
- a system+user prompt to apply,
- and OpenAI-compatible API credentials,

run the prompt over every row in parallel (via ``asyncio.Semaphore``)
and write a new parquet with two extra columns: ``<output_col>``
(the rewritten code) and ``<thinking_col>`` (the raw model reply,
useful for debugging / chain-of-thought).

The OpenAI Python client is used in *compatible mode*, so this works
with any provider that exposes an OpenAI-style API: DashScope, OpenAI,
Together, Fireworks, etc.

Used by ``reorder.py`` and ``add_comments.py``.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio


# ===========================================================================
# Code-fence parsing
# ===========================================================================

_FENCE_RE = re.compile(r"```(?:[^\n`]*)\n(.*?)```", flags=re.DOTALL)


def extract_thinking_and_code(reply: str) -> Tuple[str, Optional[str]]:
    """Split a model reply into (full_text, code_inside_fences_or_None).

    The fenced-code regex captures everything between the opening
    ``` and the closing ```, so all we have to do here is trim
    surrounding whitespace.
    """
    if not isinstance(reply, str):
        return "", None
    m = _FENCE_RE.search(reply)
    if m:
        code = m.group(1).strip()
        return reply, code if code else None
    return reply.strip(), None


# ===========================================================================
# Task spec + runner
# ===========================================================================

@dataclass
class RewriteTask:
    """Configuration for an LLM-based code-rewrite job."""

    # Prompts.
    system_prompt: str
    user_prompt: str

    # Column names.
    input_col: str           # column to rewrite
    output_col: str          # column for the rewritten code
    thinking_col: str        # column for the raw reply

    # API settings (provider, model, base url, key).
    base_url: str
    model_id: str
    api_key: str = ""        # optional; can also be picked up from env
    extra_body: Dict[str, Any] = field(default_factory=dict)

    # Run-time tuning.
    max_concurrency: int = 128
    max_retries: int = 3
    api_timeout: int = 300

    # Message formatting: how to embed the source into the user message.
    # ``{prompt}`` is the user_prompt; ``{code}`` is the row's source.
    user_template: str = (
        "{prompt}\n\n-----BEGIN_TIKZ_CODE-----\n{code}\n-----END_TIKZ_CODE-----"
    )


async def _process_row(
    row: Dict[str, Any], task: RewriteTask,
    client: AsyncOpenAI, sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single row; never raises (returns ``failed=True`` on giving up)."""
    src = row.get(task.input_col, "") or ""
    user_msg = task.user_template.format(prompt=task.user_prompt, code=src)
    messages = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": user_msg},
    ]

    async with sem:
        for attempt in range(1, task.max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=task.model_id,
                    messages=messages,
                    timeout=task.api_timeout,
                    extra_body=task.extra_body or None,
                )
                reply = (resp.choices[0].message.content or "").strip()
                thinking, code = extract_thinking_and_code(reply)
                return {
                    task.output_col: code if code else src,
                    task.thinking_col: thinking,
                    "failed": code is None,
                }
            except Exception as exc:  # noqa: BLE001
                print(f"[API error] {exc} | retry {attempt}/{task.max_retries}")
                await asyncio.sleep(min(2 * attempt, 10))

    return {
        task.output_col: src,
        task.thinking_col: "[model_failed_keep_original]",
        "failed": True,
    }


async def run_async(
    input_parquet: str,
    output_parquet: str,
    task: RewriteTask,
    drop_failed: bool = True,
) -> None:
    """Apply ``task`` to every row of ``input_parquet`` and write a new parquet."""
    print(f"Loading dataset from {input_parquet}")
    ds = load_dataset("parquet", data_files=input_parquet, split="train")
    if task.input_col not in ds.column_names:
        raise SystemExit(
            f"Column '{task.input_col}' not in parquet. "
            f"Available: {ds.column_names}"
        )

    api_key = task.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "No API key provided. Set --api-key, RewriteTask.api_key, or "
            "the OPENAI_API_KEY environment variable."
        )

    client = AsyncOpenAI(base_url=task.base_url, api_key=api_key)
    sem = asyncio.Semaphore(task.max_concurrency)

    print(
        f"Processing {len(ds)} rows with concurrency={task.max_concurrency}, "
        f"model={task.model_id}, base_url={task.base_url}"
    )
    coros = [_process_row(ds[i], task, client, sem) for i in range(len(ds))]
    results: List[Dict[str, Any]] = await tqdm_asyncio.gather(*coros)

    ds = ds.add_column(task.output_col,   [r[task.output_col]   for r in results])
    ds = ds.add_column(task.thinking_col, [r[task.thinking_col] for r in results])
    ds = ds.add_column("_rewrite_failed", [r["failed"]          for r in results])

    n_total = len(ds)
    if drop_failed:
        ds = ds.filter(lambda ex: not ex["_rewrite_failed"])
    ds = ds.remove_columns(["_rewrite_failed"])
    n_kept = len(ds)
    print(
        f"Total {n_total}, kept {n_kept} "
        f"({n_kept / max(n_total, 1):.1%})"
    )

    ds.to_parquet(output_parquet)
    print(f"Saved -> {output_parquet}")


def run(
    input_parquet: str,
    output_parquet: str,
    task: RewriteTask,
    drop_failed: bool = True,
) -> None:
    """Synchronous wrapper around :func:`run_async`."""
    asyncio.run(run_async(input_parquet, output_parquet, task, drop_failed))


# ===========================================================================
# CLI helpers shared by reorder.py / add_comments.py
# ===========================================================================

def add_common_cli_args(parser) -> None:
    """Attach the standard CLI flags to an ``argparse.ArgumentParser``."""
    parser.add_argument("--input", required=True, help="Input parquet")
    parser.add_argument("--output", required=True, help="Output parquet")
    parser.add_argument(
        "--api-key",
        help="API key. Defaults to env OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--model-id", help="Override the default model id")
    parser.add_argument(
        "--concurrency", type=int, default=128,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Retries per row on error",
    )
    parser.add_argument(
        "--api-timeout", type=int, default=300, help="Per-call timeout (s)",
    )
    parser.add_argument(
        "--keep-failed", action="store_true",
        help="Keep failed rows in the output (default: drop them)",
    )
