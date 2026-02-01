"""
Reorder TikZ source code via an LLM.

Asks the model to rewrite a piece of TikZ code in a more natural plotting
order (handling forward references, relative-position recomputation,
and unused tikzset cleanup), then saves the rewritten code to a new
column ``new_code`` and the model's thinking to ``thinking``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from llm_runner import RewriteTask, add_common_cli_args, run


SYSTEM_PROMPT = "You are an expert LaTeX/TikZ engineer."

USER_PROMPT = (
    "You are required to change the TikZ code order if you find the plotting "
    "order is not natural or not intuitive. However, do not impact the "
    "rendered image content if you choose to alter the code. Also, pay "
    "attention to variable and macro declarations when you adjust code. "
    "We can not reference any variable or macro before it is defined. "
    "I understand that when you change the order, you may need to change "
    "the relative position if the referred variable is moved to after. "
    "You need to carefully think about the position expression "
    "transformation. Specifically, you need to understand the coordinate "
    "rules of TikZ. For example, when you define an object at (x,y), (x,y) "
    "refers to the center of the object. However, when you use relative "
    "positions like 'left' or 'right' of an object, it refers to the edge "
    "of the object. Think before generating code. Return your thinking "
    "briefly with output code. If the code defines some common styles "
    "using tikzset or tikzestyle, try to check if the defined sets and "
    "styles are used. If some style sets and styles are not used, remove "
    "them, just as you would remove unused imports."
)

DEFAULT_MODEL_ID = "qwen3-coder-plus"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reorder TikZ code via an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--input-col", default="code",
        help="Source column with original TikZ code",
    )
    parser.add_argument(
        "--output-col", default="new_code",
        help="Target column for the reordered code",
    )
    parser.add_argument(
        "--thinking-col", default="thinking",
        help="Column for the raw model reply",
    )
    args = parser.parse_args(argv)

    task = RewriteTask(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        input_col=args.input_col,
        output_col=args.output_col,
        thinking_col=args.thinking_col,
        base_url=args.base_url,
        model_id=args.model_id or DEFAULT_MODEL_ID,
        api_key=args.api_key or "",
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
    )
    run(
        input_parquet=args.input,
        output_parquet=args.output,
        task=task,
        drop_failed=not args.keep_failed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
