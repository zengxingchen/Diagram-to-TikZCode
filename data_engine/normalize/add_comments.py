"""
Add inline comments to TikZ source code via an LLM.

Asks the model to (1) describe the rendered image, (2) reason about
how it would write the code from scratch, (3) annotate the **original**
code with comments without altering anything else. The annotated code
is saved to ``commented_code`` and the model's thinking to
``comment_thinking``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from llm_runner import RewriteTask, add_common_cli_args, run


SYSTEM_PROMPT = "You are an expert LaTeX/TikZ engineer."

USER_PROMPT = (
    "You are an expert in Tikz. I will give you a piece of tikz code. "
    "Briefly depict what the rendered image looks like in less than "
    "100 words, wrapped within <Depict> <\\Depict> block. "
    "Then imaging you are looking at the image, forgetting about the "
    "original code, and actively think how you will write your own "
    "code to produce the image. Please compress your raw thoughts to "
    "up to 600 words, while preserving your original language style, "
    "tone, and personality. The raw thoughts should be returned within "
    "<Think><\\Think> block. Finally, annotate the whole original code. "
    "Do not change any original code. Just add comments to it. "
    "Return the whole annotated original code inside triple backticks ```."
)

DEFAULT_MODEL_ID = "qwen-plus-2025-07-28"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Annotate TikZ code with inline comments via an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--input-col", default="new_code",
        help="Source column with TikZ code to annotate",
    )
    parser.add_argument(
        "--output-col", default="commented_code",
        help="Target column for the annotated code",
    )
    parser.add_argument(
        "--thinking-col", default="comment_thinking",
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
        # Disable thinking mode for the comment task: this model emits a
        # structured response that already encodes its reasoning.
        extra_body={"enable_thinking": False},
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        # The comment task slots the code straight into the prompt body.
        user_template="{prompt}\n\n {code}\n",
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
