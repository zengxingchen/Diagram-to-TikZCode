"""
LaTeX compilation backend for the TikZ Import Optimizer.

Two layers, both used by ``import_optimizer``:

- ``tex2img(code, ...)``        — Compile a LaTeX string to a PDF + PNG
                                  using ``latexmk`` and a fallback chain
                                  of engines (``pdflatex`` → ``lualatex``
                                  → ``xelatex``).
- ``LaTeXCompiler``             — A thin wrapper that adapts ``tex2img``
                                  into a ``test_compile(source) ->
                                  (ok, error, elapsed_seconds)`` oracle,
                                  which is what the optimizer actually
                                  needs.

System dependencies: ``latexmk``, a LaTeX engine, ``poppler``
(``pdftoppm`` / ``pdfinfo``), and ``ghostscript`` (used by
``pdfCropMargins``).

Python dependencies: ``pymupdf``, ``pdf2image``, ``pdfCropMargins``,
``Pillow``.
"""

from __future__ import annotations

import os
import subprocess
import time
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Optional, Tuple

import pymupdf
from pdf2image.pdf2image import convert_from_path
from pdfCropMargins import crop
from PIL import ImageOps


# ===========================================================================
# tex2img: LaTeX string -> PDF bytes + PNG bytes
# ===========================================================================

_ENGINES = ("pdflatex", "lualatex", "xelatex")


def tex2img(
    code: str,
    size: int = 384,
    timeout: int = 120,
    expand_to_square: bool = True,
    verbose: bool = True,
) -> Dict[str, bytes]:
    """Compile ``code`` and return ``{"pdf": <bytes>, "image": <png bytes>}``.

    Tries ``pdflatex`` first, falling back to ``lualatex`` then
    ``xelatex``. Raises ``ValueError`` on failure.
    """
    codelines = code.split("\n")
    # Suppress page headers and footers so cropping works reliably.
    codelines.insert(
        1,
        r"{cmd}\AtBeginDocument{{{cmd}}}".format(
            cmd=r"\thispagestyle{empty}\pagestyle{empty}"
        ),
    )

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    def _try_compile(file_stem: str, cwd: str) -> str:
        """Compile ``file_stem.tex`` and return path to the produced PDF."""
        # Some classes need a .bbl to exist even when bibtex is disabled.
        open(f"{file_stem}.bbl", "a").close()

        for engine in _ENGINES:
            try:
                result = subprocess.run(
                    [
                        "latexmk", "-nobibtex", "-norc",
                        "-interaction=nonstopmode", f"-{engine}", file_stem,
                    ],
                    cwd=cwd,
                    capture_output=True, text=True, timeout=timeout,
                )
                if result.returncode == 0:
                    if engine != _ENGINES[0]:
                        _log(f"{_ENGINES[0]} failed, but {engine} succeeded")
                    return f"{file_stem}.pdf"
                if engine == _ENGINES[-1]:
                    _log("All engines failed.")
            except subprocess.TimeoutExpired:
                if engine == _ENGINES[-1]:
                    _log("All engines timed out")
            except Exception as exc:  # noqa: BLE001
                if engine == _ENGINES[-1]:
                    _log(f"All engines failed with exception: {exc}")
        raise ValueError("Couldn't compile latex source with any engine.")

    with TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "temp.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(codelines))

        pdf_path = _try_compile(tex_path[:-4], tmpdir)  # strip ".tex"

        # Keep only the last page (covers multi-page outputs).
        doc = pymupdf.open(pdf_path)
        doc.select([len(doc) - 1])
        doc.saveIncr()

        # Crop whitespace.
        cropped = tex_path.replace(".tex", "-cropped.pdf")
        crop(["-c", "gb", "-p", "0", "-a", "-1", "-o", cropped, pdf_path],
             quiet=True)

        # PDF -> PNG.
        image = convert_from_path(cropped, size=size, single_file=True)[0]
        if expand_to_square:
            image = ImageOps.pad(image, (size, size), color="white")

        if image.getcolors(1) is not None:
            raise ValueError("Provided code compiled to an empty image.")

        with open(cropped, "rb") as f:
            pdf_bytes = f.read()
        if not isinstance(pdf_bytes, bytes) or len(pdf_bytes) < 1000:
            raise ValueError("Cropped PDF is invalid or too small")

        buf = BytesIO()
        image.save(buf, format="PNG")
        return {"pdf": pdf_bytes, "image": buf.getvalue()}


# ===========================================================================
# LaTeXCompiler: "does this compile?" oracle for the optimizer
# ===========================================================================

class LaTeXCompiler:
    """Adapts a ``tex2img``-style callable into a compile-only oracle."""

    def __init__(
        self,
        tex2img_fn: Callable[..., Any] = tex2img,
        timeout: int = 30,
    ) -> None:
        self._tex2img = tex2img_fn
        self.timeout = timeout

    def test_compile(
        self, content: str
    ) -> Tuple[bool, Optional[str], float]:
        """Return ``(ok, error_message, elapsed_seconds)``."""
        start = time.time()
        try:
            self._tex2img(
                content, size=384, timeout=self.timeout,
                expand_to_square=False, verbose=False,
            )
            return True, None, time.time() - start
        except Exception as exc:  # noqa: BLE001
            return False, str(exc), time.time() - start
