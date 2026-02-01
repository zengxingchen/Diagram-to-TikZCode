"""
TikZ Package Optimizer
======================

Iteratively prunes redundant ``\\usepackage`` and ``\\usetikzlibrary``
declarations (and optionally rewrites ``\\documentclass`` to
``standalone``) from LaTeX / TikZ source files. The optimizer compiles
candidate sources via :class:`latex_compiler.LaTeXCompiler` and only
keeps dependencies that compilation actually requires.

Architecture
------------
- ``LaTeXParser``    Token-level extraction of packages, libraries and
                     ``\\documentclass`` using ``pygments`` ``TexLexer``.
- ``LaTeXEditor``    In-place edits: remove a package / library, drop
                     ``\\documentclass`` options, convert to
                     ``standalone``, de-duplicate. Preserves comments
                     and surrounding source.
- ``LaTeXCompiler``  Lives in ``latex_compiler.py``; wraps the
                     LaTeX-to-PDF pipeline as a
                     ``test_compile(source) -> (ok, err, secs)`` oracle.
- ``TikZOptimizer``  Drives optimization: fast path -> Pass 1 (greedy)
                     -> Pass 2 (validation) -> library pass; optional
                     ``\\documentclass`` rewrite first.

Usage
-----
::

    python import_optimizer.py --input figure.tex --output out_dir
    python import_optimizer.py --input tex_dir/  --output out_dir \\
        --workers 8 --standalone

Programmatic::

    from import_optimizer import run
    results = run(input_path="tex_dir/", output_dir="out_dir", workers=8)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pygments.lexers.markup import TexLexer
from pygments.token import Token

from latex_compiler import LaTeXCompiler

logger = logging.getLogger("import_optimizer")

# A pygments token: (token_type, value).
Tokens = List[Tuple[object, str]]

# Core packages that the fast path always tries to keep.
CORE_PACKAGES = frozenset({"tikz", "amssymb", "amsmath"})

# Documentclass -> packages that the class normally provides implicitly.
IMPLICIT_PACKAGES_MAP: dict = {
    "amsart": ["amsmath", "amssymb"],
}

_MULTI_BLANK_RE = re.compile(r"\n{3,}")


# ===========================================================================
# Token-level utilities
# ===========================================================================

def extract_balanced(
    tokens: Tokens, start_idx: int, opener: str, closer: str
) -> Optional[Tuple[str, int]]:
    """Extract content of a balanced ``opener...closer`` block.

    Returns ``(content, end_index)`` past the closing delimiter, or
    ``None`` if no opening delimiter is found at ``start_idx`` (after
    skipping whitespace) or if the block is empty.
    """
    i = start_idx
    while i < len(tokens) and tokens[i][1].strip() == "":
        i += 1
    if i >= len(tokens):
        return None

    value = tokens[i][1].strip()
    if not value.startswith(opener):
        return None

    # Whole "opener...closer" lives in one token.
    if (
        value.count(opener) == value.count(closer)
        and value.endswith(closer)
        and value.count(opener) >= 1
    ):
        content = value[1:-1].strip()
        return (content, i + 1) if content else None

    parts: List[str] = []
    depth = value.count(opener) - value.count(closer)

    if value == opener:
        i += 1
    elif closer in value and depth <= 0:
        # Opener-bearing token also embeds the closer; split.
        sp, ep = value.index(opener), value.rfind(closer)
        content = value[sp + 1 : ep].strip()
        return (content, i + 1) if content else None
    else:
        after = value[value.index(opener) + 1 :]
        if after:
            parts.append(after)
        i += 1

    while i < len(tokens) and depth > 0:
        _, value = tokens[i]
        depth += value.count(opener) - value.count(closer)
        if depth > 0:
            parts.append(value)
        elif closer in value:
            parts.append(value[: value.rfind(closer)])
        i += 1

    content = " ".join("".join(parts).split())
    return (content, i) if content else None


def _collect_optional_block(tokens: Tokens, i: int) -> Tuple[Tokens, int]:
    """Collect raw tokens of a ``[...]`` block at ``i``.

    Returns ``(collected_tokens, new_index)``. If no ``[`` is at ``i``,
    returns ``([], i)`` unchanged.
    """
    if i >= len(tokens) or "[" not in tokens[i][1]:
        return [], i
    collected: Tokens = []
    depth = 0
    while i < len(tokens):
        t_type, t_value = tokens[i]
        collected.append((t_type, t_value))
        depth += t_value.count("[") - t_value.count("]")
        i += 1
        if depth <= 0:
            break
    return collected, i


def _skip_optional_block(tokens: Tokens, i: int) -> int:
    """Advance past a ``[...]`` block at ``i`` without keeping it."""
    return _collect_optional_block(tokens, i)[1]


def _consume_rest_of_line(
    tokens: Tokens, i: int, out: Optional[List[str]] = None
) -> int:
    """Eat trailing whitespace / comment up to and including one newline.

    When deleting a whole ``\\usepackage`` / ``\\usetikzlibrary`` line we
    also eat anything that follows it on the same source line (a trailing
    ``% comment`` and the newline) to avoid orphan fragments.

    Tokens that span multiple lines (e.g. ``"\\nfoo\\n"``) are *partially*
    consumed: the leading portion up to the first ``\\n`` is eaten, the
    remainder is pushed to ``out`` so it stays in the source.
    """
    while i < len(tokens):
        token_type, value = tokens[i]
        if token_type in Token.Comment or value.strip() == "":
            if "\n" in value:
                return i + 1
            i += 1
            continue
        if "\n" in value:
            head, _, tail = value.partition("\n")
            if head.strip() == "":
                if out is not None and tail:
                    out.append(tail)
                return i + 1
        return i  # real code on this line; don't eat it
    return i


def _collapse_blank_lines(s: str) -> str:
    """Collapse runs of 3+ consecutive newlines to a single blank line."""
    return _MULTI_BLANK_RE.sub("\n\n", s)


# ===========================================================================
# 1. LaTeX Parser
# ===========================================================================

class LaTeXParser:
    """Token-level parser for LaTeX preamble elements."""

    def __init__(self) -> None:
        self.lexer = TexLexer()

    def extract_packages_and_libraries(
        self, content: str
    ) -> Tuple[List[str], List[str]]:
        """Return ordered, de-duplicated ``(packages, libraries)``."""
        packages: List[str] = []
        libraries: List[str] = []
        tokens = list(self.lexer.get_tokens(content))
        i = 0
        while i < len(tokens):
            token_type, value = tokens[i]
            if token_type in Token.Comment:
                i += 1
                continue

            if token_type in Token.Keyword and value in (
                r"\usepackage", r"\usetikzlibrary"
            ):
                target = packages if value == r"\usepackage" else libraries
                i += 1
                if value == r"\usepackage":
                    i = _skip_optional_block(tokens, i)
                if i < len(tokens):
                    extracted = extract_balanced(tokens, i, "{", "}")
                    if extracted is not None:
                        body, i = extracted
                        for name in (n.strip() for n in body.split(",")):
                            if name and name not in target:
                                target.append(name)
                        continue
            i += 1

        return packages, libraries

    def extract_documentclass(
        self, content: str
    ) -> Optional[Tuple[str, List[str], str]]:
        """Return ``(keyword, options, class_name)`` or ``None``."""
        tokens = list(self.lexer.get_tokens(content))
        i = 0
        while i < len(tokens):
            token_type, value = tokens[i]
            if token_type in Token.Keyword and value == r"\documentclass":
                i += 1
                options: List[str] = []
                if i < len(tokens) and "[" in tokens[i][1]:
                    extracted = extract_balanced(tokens, i, "[", "]")
                    if extracted is not None:
                        opts_str, i = extracted
                        options = [
                            o.strip() for o in opts_str.split(",") if o.strip()
                        ]
                if i < len(tokens):
                    extracted = extract_balanced(tokens, i, "{", "}")
                    if extracted is not None:
                        return value, options, extracted[0].strip()
            i += 1
        return None


# ===========================================================================
# 2. LaTeX Editor
# ===========================================================================

# Filter: given a list of names found in a single declaration, return the
# subset that should remain. Used to build remove / dedup variants.
NameFilter = Callable[[List[str]], List[str]]


class LaTeXEditor:
    """Token-level edits for LaTeX source."""

    def __init__(self) -> None:
        self.lexer = TexLexer()
        self.parser = LaTeXParser()

    # ----- generic declaration rewrite --------------------------------------

    def _rewrite_declarations(
        self,
        content: str,
        keyword: str,
        with_options: bool,
        name_filter: NameFilter,
    ) -> str:
        """Walk ``content`` and rewrite every declaration of ``keyword``.

        For each ``keyword{name1, name2, ...}`` declaration the names are
        passed to ``name_filter``, and the declaration is either kept
        (with the surviving names) or dropped together with its trailing
        comment / newline.
        """
        tokens = list(self.lexer.get_tokens(content))
        out: List[str] = []
        i = 0
        while i < len(tokens):
            token_type, value = tokens[i]
            if token_type in Token.Keyword and value == keyword:
                i += 1
                opts_tokens, i = (
                    _collect_optional_block(tokens, i)
                    if with_options else ([], i)
                )

                if i < len(tokens):
                    extracted = extract_balanced(tokens, i, "{", "}")
                    if extracted is not None:
                        body, end_idx = extracted
                        kept = name_filter(
                            [n.strip() for n in body.split(",")]
                        )
                        if kept:
                            out.append(keyword)
                            out.extend(t[1] for t in opts_tokens)
                            out.append("{" + ", ".join(kept) + "}")
                            i = end_idx
                        else:
                            i = _consume_rest_of_line(tokens, end_idx, out)
                        continue
                    # Bare keyword without {...}: drop the whole line.
                    i = _consume_rest_of_line(tokens, i, out)
                    continue

            out.append(value)
            i += 1
        return _collapse_blank_lines("".join(out))

    # ----- single-declaration removal ---------------------------------------

    def remove_package(self, content: str, package: str) -> str:
        """Drop ``package`` from every ``\\usepackage`` declaration."""
        return self._rewrite_declarations(
            content, r"\usepackage", with_options=True,
            name_filter=lambda names: [n for n in names if n and n != package],
        )

    def remove_library(self, content: str, library: str) -> str:
        """Drop ``library`` from every ``\\usetikzlibrary`` declaration."""
        return self._rewrite_declarations(
            content, r"\usetikzlibrary", with_options=False,
            name_filter=lambda names: [n for n in names if n and n != library],
        )

    # ----- de-duplication ---------------------------------------------------

    @staticmethod
    def _dedup_filter(seen: set) -> NameFilter:
        def _filter(names: List[str]) -> List[str]:
            unique = []
            for n in names:
                if n and n not in seen:
                    seen.add(n)
                    unique.append(n)
            return unique
        return _filter

    def remove_duplicates(self, content: str) -> str:
        """Collapse repeated ``\\usepackage`` and ``\\usetikzlibrary``."""
        content = self._rewrite_declarations(
            content, r"\usepackage", with_options=True,
            name_filter=self._dedup_filter(set()),
        )
        content = self._rewrite_declarations(
            content, r"\usetikzlibrary", with_options=False,
            name_filter=self._dedup_filter(set()),
        )
        return content

    # ----- documentclass option removal -------------------------------------

    def remove_documentclass_option(
        self, content: str, option_to_remove: str
    ) -> str:
        """Remove a single option from ``\\documentclass[...]``."""
        info = self.parser.extract_documentclass(content)
        if info is None:
            return content
        _, options, class_name = info
        remaining = [opt for opt in options if opt != option_to_remove]
        new_docclass = self._build_documentclass(class_name, remaining)
        return self._replace_documentclass(content, new_docclass)

    @staticmethod
    def _build_documentclass(class_name: str, options: List[str]) -> str:
        if options:
            return r"\documentclass[" + ",".join(options) + "]{" + class_name + "}"
        return r"\documentclass{" + class_name + "}"

    @staticmethod
    def _build_standalone(border: str) -> str:
        return r"\documentclass[border=" + border + r"]{standalone}"

    def _replace_documentclass(self, content: str, replacement: str) -> str:
        """Replace the first ``\\documentclass`` declaration verbatim."""
        tokens = list(self.lexer.get_tokens(content))
        out: List[str] = []
        i = 0
        while i < len(tokens):
            token_type, value = tokens[i]
            if token_type in Token.Keyword and value == r"\documentclass":
                out.append(replacement)
                i += 1
                i = _skip_optional_block(tokens, i)
                if i < len(tokens):
                    extracted = extract_balanced(tokens, i, "{", "}")
                    if extracted is not None:
                        i = extracted[1]
                        continue
            out.append(value)
            i += 1
        return "".join(out)

    # ----- documentclass -> standalone --------------------------------------

    def convert_to_standalone(
        self,
        content: str,
        border: str = "2mm",
        compiler: Optional["LaTeXCompiler"] = None,
    ) -> str:
        r"""Rewrite ``\documentclass[...]{cls}`` to ``standalone``.

        * Fast path: try removing all options at once.
        * Otherwise: test each option individually; necessary ones are
          migrated to ``\usepackage`` lines.
        * Add implicit packages required by some classes.
        """
        info = self.parser.extract_documentclass(content)
        if info is None:
            return self._build_standalone(border) + "\n" + content

        _, options, class_name = info
        new_docclass = self._build_standalone(border)

        if options and compiler is not None:
            necessary = self._migrate_documentclass_options(
                content, options, compiler
            )
            if necessary:
                logger.info(
                    "   Converting necessary options to packages: %s",
                    necessary,
                )
                new_docclass += "\n" + "\n".join(
                    r"\usepackage{" + opt + "}" for opt in necessary
                )

        implicit = IMPLICIT_PACKAGES_MAP.get(class_name.lower(), [])
        if implicit:
            existing, _ = self.parser.extract_packages_and_libraries(content)
            to_add = [p for p in implicit if p not in existing]
            if to_add:
                logger.info(
                    "   Adding implicit packages from '%s': %s",
                    class_name, to_add,
                )
                new_docclass += "\n" + "\n".join(
                    r"\usepackage{" + p + "}" for p in to_add
                )

        return self._replace_documentclass(content, new_docclass)

    def _migrate_documentclass_options(
        self,
        content: str,
        options: List[str],
        compiler: "LaTeXCompiler",
    ) -> List[str]:
        """Return the subset of ``options`` that must become ``\\usepackage``."""
        logger.info(
            "   Fast path: testing removal of all %d documentclass "
            "options: %s", len(options), options,
        )
        # Fast path: drop every option at once.
        test_content = content
        for option in options:
            test_content = self.remove_documentclass_option(
                test_content, option
            )
        ok, _, dt = compiler.test_compile(test_content)
        if ok:
            logger.info(
                "   Fast path SUCCESS! All options removable (%.1fs)", dt
            )
            return []

        logger.info(
            "   Fast path failed (%.1fs), testing each option individually...",
            dt,
        )
        necessary: List[str] = []
        for option in options:
            test_content = self.remove_documentclass_option(content, option)
            ok, _, dt = compiler.test_compile(test_content)
            logger.info(
                "      option '%s': %s (%.1fs)",
                option, "not needed" if ok else "needed", dt,
            )
            if not ok:
                necessary.append(option)
        return necessary


# ===========================================================================
# 3. Result type and TikZ Optimizer
# ===========================================================================

@dataclass
class OptimizationResult:
    """Outcome of optimizing a single ``.tex`` file."""

    file: str
    status: str  # "success" | "error"
    error: Optional[str] = None
    original_packages: List[str] = field(default_factory=list)
    original_libraries: List[str] = field(default_factory=list)
    removable_packages: List[str] = field(default_factory=list)
    removable_libraries: List[str] = field(default_factory=list)
    optimized_content: Optional[str] = None
    output_file: Optional[str] = None
    compilation_success: Optional[bool] = None
    compilation_error: Optional[str] = None


class TikZOptimizer:
    """Coordinator: parser + editor + compiler + iterative removal."""

    def __init__(
        self, compiler: Optional[LaTeXCompiler] = None, timeout: int = 30,
    ) -> None:
        self.parser = LaTeXParser()
        self.editor = LaTeXEditor()
        self.compiler = compiler if compiler is not None else LaTeXCompiler(
            timeout=timeout
        )

    # ----- core algorithm ---------------------------------------------------

    def _try_drop(
        self, current: str, name: str, remove_fn: Callable[[str, str], str]
    ) -> Tuple[bool, str, float]:
        """Try removing ``name`` via ``remove_fn``; return (ok, candidate, secs)."""
        modified = remove_fn(current, name)
        ok, _, dt = self.compiler.test_compile(modified)
        return ok, modified, dt

    def _greedy_drop(
        self,
        names: List[str],
        current: str,
        remove_fn: Callable[[str, str], str],
        retry_failures: bool,
    ) -> Tuple[List[str], List[str], str]:
        """One greedy pass: try to drop each name; return (removable, kept, current).

        If ``retry_failures`` is True the kept ones are returned for a
        second pass; otherwise an empty kept list is returned.
        """
        removable: List[str] = []
        kept: List[str] = []
        for name in names:
            ok, modified, dt = self._try_drop(current, name, remove_fn)
            if ok:
                logger.info("    %-30s removed (%.1fs)", name, dt)
                removable.append(name)
                current = modified
            else:
                logger.info("    %-30s needed  (%.1fs)", name, dt)
                if retry_failures:
                    kept.append(name)
        return removable, kept, current

    def _incremental_removal(
        self, content: str, packages: List[str], libraries: List[str]
    ) -> Tuple[List[str], List[str], str]:
        """Return ``(removable_packages, removable_libraries, optimized)``."""
        removable_packages: List[str] = []
        current = content

        # ----- Fast path: try to drop all non-core packages at once -----
        non_core = [p for p in packages if p not in CORE_PACKAGES]
        existing_core = [p for p in packages if p in CORE_PACKAGES]
        packages_to_test = packages
        fast_path_ok = False

        if non_core:
            logger.info(
                "  Fast path: aggressive removal of %d non-core packages "
                "(keeping only %s)", len(non_core), sorted(CORE_PACKAGES),
            )
            test_content = current
            for pkg in non_core:
                test_content = self.editor.remove_package(test_content, pkg)
            ok, _, dt = self.compiler.test_compile(test_content)
            if ok:
                logger.info(
                    "  Fast path SUCCESS (%.1fs); now testing core: %s",
                    dt, existing_core,
                )
                removable_packages.extend(non_core)
                current = test_content
                packages_to_test = existing_core
                fast_path_ok = True
            else:
                logger.info(
                    "  Fast path failed (%.1fs); falling back to greedy", dt,
                )
        else:
            logger.info("  Fast path skipped (only core packages present)")

        # ----- Pass 1: greedy -----
        logger.info(
            "  Pass 1: %s", "core packages" if fast_path_ok else "greedy removal",
        )
        removed1, possibly_needed, current = self._greedy_drop(
            packages_to_test, current, self.editor.remove_package,
            retry_failures=True,
        )
        removable_packages.extend(removed1)

        # ----- Pass 2: validate "needed" packages against the smaller state -
        if possibly_needed:
            logger.info("  Pass 2: validating 'needed' packages")
            removed2, _, current = self._greedy_drop(
                possibly_needed, current, self.editor.remove_package,
                retry_failures=False,
            )
            removable_packages.extend(removed2)

        # ----- Library pass -----
        if libraries:
            logger.info("  Library pass")
        removable_libraries, _, current = self._greedy_drop(
            libraries, current, self.editor.remove_library,
            retry_failures=False,
        )

        return removable_packages, removable_libraries, current

    # ----- per-file pipeline ------------------------------------------------

    def optimize_packages(
        self,
        file_path: str,
        use_standalone: bool = False,
        standalone_border: str = "2mm",
    ) -> OptimizationResult:
        """Run the full optimization for one file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            original = f.read()

        logger.info("Testing original file: %s", file_path)
        ok, error, _ = self.compiler.test_compile(original)
        if not ok:
            return OptimizationResult(
                file=file_path, status="error",
                error=f"Original file does not compile: {error}",
                optimized_content=original,
            )

        content = original
        if use_standalone:
            logger.info("Converting documentclass to standalone...")
            content = self.editor.convert_to_standalone(
                original, border=standalone_border, compiler=self.compiler,
            )

        packages, libraries = self.parser.extract_packages_and_libraries(content)
        logger.info(
            "Found %d packages and %d libraries", len(packages), len(libraries),
        )
        logger.debug("Packages: %s", packages)
        logger.debug("Libraries: %s", libraries)
        logger.info("Iterative package removal...")

        rem_pkgs, rem_libs, optimized = self._incremental_removal(
            content, packages, libraries
        )
        return OptimizationResult(
            file=file_path, status="success",
            original_packages=packages, original_libraries=libraries,
            removable_packages=rem_pkgs, removable_libraries=rem_libs,
            optimized_content=optimized,
        )

    def create_optimized_file(
        self, output_path: str, optimized_content: str
    ) -> Tuple[str, bool, Optional[str]]:
        """De-dup, verify-compile and write. Returns ``(content, ok, err)``."""
        content = self.editor.remove_duplicates(optimized_content)
        logger.info("   Verifying optimized file compiles...")
        ok, error, dt = self.compiler.test_compile(content)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        if ok:
            logger.info("   Verified compilation OK (%.1fs)", dt)
            return content, True, None
        logger.warning("   WARNING: optimized file does not compile (%s)", error)
        return content, False, error


# ===========================================================================
# Multiprocessing worker
# ===========================================================================

@dataclass
class _WorkerArgs:
    file_path: str
    timeout: int
    use_standalone: bool
    standalone_border: str
    output_dir: Optional[str]
    input_path: str
    log_level: int


def _resolve_output_path(
    input_path: str, file_path: str, output_dir: str
) -> Path:
    """Decide the output file path, preserving directory layout."""
    out = Path(output_dir)
    if os.path.isdir(input_path):
        return out / Path(file_path).relative_to(Path(input_path))
    return out / Path(file_path).name


def _process_single_file(args: _WorkerArgs) -> OptimizationResult:
    # Workers are spawned processes (on macOS / Windows); reconfigure logging.
    logging.basicConfig(level=args.log_level, format="%(message)s")

    optimizer = TikZOptimizer(timeout=args.timeout)
    result = optimizer.optimize_packages(
        args.file_path,
        use_standalone=args.use_standalone,
        standalone_border=args.standalone_border,
    )

    if result.status == "success" and args.output_dir:
        out_file = _resolve_output_path(
            args.input_path, args.file_path, args.output_dir
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        _, ok, err = optimizer.create_optimized_file(
            str(out_file), result.optimized_content or ""
        )
        result.output_file = str(out_file)
        result.compilation_success = ok
        result.compilation_error = err
    elif result.status != "success":
        result.compilation_success = False
        result.compilation_error = result.error

    return result


# ===========================================================================
# Programmatic API
# ===========================================================================

def _collect_tex_files(
    input_p: Path, max_files: Optional[int]
) -> List[Path]:
    if input_p.is_file():
        return [input_p]
    if input_p.is_dir():
        files = sorted(input_p.glob("**/*.tex"))
        return files[:max_files] if max_files else files
    raise FileNotFoundError(
        f"Input is neither a file nor a directory: {input_p}"
    )


def run(
    input_path: str,
    output_dir: str,
    timeout: int = 30,
    workers: int = 4,
    max_files: Optional[int] = None,
    use_standalone: bool = False,
    standalone_border: str = "2mm",
) -> List[OptimizationResult]:
    """Optimize one file or every ``.tex`` under a directory.

    Returns one :class:`OptimizationResult` per processed file. Logging
    is controlled via the ``import_optimizer`` logger (set to
    ``WARNING`` for quiet operation).
    """
    input_p = Path(input_path)
    tex_files = _collect_tex_files(input_p, max_files)
    if not tex_files:
        logger.warning("No .tex files found at %s", input_path)
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_workers = max(1, min(workers, len(tex_files)))
    logger.info(
        "Optimizing %d file(s)  (timeout=%ds, workers=%d, standalone=%s)",
        len(tex_files), timeout, n_workers, use_standalone,
    )

    log_level = logging.getLogger().getEffectiveLevel()
    arg_list = [
        _WorkerArgs(
            file_path=str(p), timeout=timeout,
            use_standalone=use_standalone,
            standalone_border=standalone_border,
            output_dir=output_dir, input_path=str(input_path),
            log_level=log_level,
        )
        for p in tex_files
    ]

    started = time.time()
    results: List[OptimizationResult] = []
    pool: Optional[Pool] = None  # type: ignore[type-arg]
    if n_workers == 1:
        iterator = (_process_single_file(a) for a in arg_list)
    else:
        pool = Pool(processes=n_workers)
        iterator = pool.imap(_process_single_file, arg_list)

    try:
        for i, (tex_file, result) in enumerate(zip(tex_files, iterator), 1):
            results.append(result)
            _log_file_result(i, len(tex_files), tex_file, result)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - started
    successful = sum(1 for r in results if r.status == "success")
    logger.info("=" * 60)
    logger.info("Files processed       : %d", len(tex_files))
    logger.info("Successful            : %d", successful)
    logger.info("Total time            : %.1fs", elapsed)
    logger.info("Output directory      : %s", output_dir)
    return results


def _log_file_result(
    idx: int, total: int, tex_file: Path, result: OptimizationResult
) -> None:
    header = f"[{idx}/{total}] {tex_file}"
    if result.status != "success":
        logger.error("%s -> %s", header, result.error)
        return

    n_removed = (
        len(result.removable_packages) + len(result.removable_libraries)
    )
    n_total = (
        len(result.original_packages) + len(result.original_libraries)
    )
    logger.info("%s -> removed %d/%d", header, n_removed, n_total)
    if result.removable_packages:
        logger.info("    packages : %s", ", ".join(result.removable_packages))
    if result.removable_libraries:
        logger.info("    libraries: %s", ", ".join(result.removable_libraries))
    if result.output_file and not result.compilation_success:
        logger.warning(
            "    saved (compilation failed) -> %s : %s",
            result.output_file, result.compilation_error,
        )
    elif result.output_file:
        logger.info("    saved -> %s", result.output_file)


# ===========================================================================
# CLI
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Iteratively remove redundant LaTeX/TikZ dependencies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="Input .tex file or directory")
    p.add_argument("--output", "-o", required=True,
                   help="Output directory for optimized files")
    p.add_argument("--timeout", "-t", type=int, default=30,
                   help="Per-compilation timeout (seconds)")
    p.add_argument("--workers", "-w", type=int, default=4,
                   help="Parallel workers (use 1 for sequential)")
    p.add_argument("--max-files", type=int,
                   help="Limit number of files processed (for debugging)")
    p.add_argument("--standalone", action="store_true",
                   help="Convert \\documentclass to standalone before optimizing")
    p.add_argument("--border", default="2mm",
                   help="Border size for standalone class")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Reduce log verbosity (warnings and summary only).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    run(
        input_path=args.input,
        output_dir=args.output,
        timeout=args.timeout,
        workers=args.workers,
        max_files=args.max_files,
        use_standalone=args.standalone,
        standalone_border=args.border,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
