# TikZ Import Optimization

A self-contained command-line tool and Python library that **prunes
redundant `\usepackage{...}` and `\usetikzlibrary{...}` declarations**
from LaTeX / TikZ source files. It works by *actually compiling*
candidate sources and only keeping dependencies that compilation
requires. Optionally rewrites `\documentclass[opts]{cls}` to
`standalone`, migrating only the options that compilation still needs.

Useful when cleaning up TikZ corpora collected from heterogeneous
sources, where authors typically add many "just in case" imports.

---

## Repository layout

This directory is **self-contained** — no cross-folder imports:

```
tikz_import_optimization/
├── README.md                        # this file
├── requirements.txt                 # Python dependencies
├── import_optimizer.py        # main optimizer (CLI + library)
├── latex_compiler.py                # tex2img + LaTeXCompiler oracle
└── examples/
    ├── original/                    # 20 real-world inputs
    └── optimized/                   # the same 20 files after optimization
```

The two `examples/{original,optimized}/` directories let you inspect
the tool's behaviour at a glance — diff any pair of files to see
exactly what was pruned.

---

## How it works

Four cooperating components:

| Component | Responsibility |
|-----------|----------------|
| `LaTeXParser` | Token-level extraction of packages, libraries and `\documentclass`, using `pygments.lexers.markup.TexLexer`. |
| `LaTeXEditor` | In-place edits: drop a package / library, drop a `\documentclass` option, convert to `standalone`, de-duplicate. Preserves comments and original surroundings. |
| `LaTeXCompiler` | Defined in `latex_compiler.py`. Wraps `tex2img` as a "does this compile?" oracle. |
| `TikZOptimizer` | Algorithm: *fast path → Pass 1 (greedy) → Pass 2 (validation) → library pass*. |

Algorithm details:

1. **Fast path** — try removing every non-core package in one shot
   (core = `tikz`, `amssymb`, `amsmath`); accept only if compilation
   succeeds.
2. **Pass 1 (greedy)** — drop one dependency at a time, keeping the
   accepted state.
3. **Pass 2 (validation)** — re-test packages that looked "needed" in
   Pass 1; greedy ordering can make a package falsely look required.
4. **Library pass** — same greedy strategy for `\usetikzlibrary`.
5. (Optional) `--standalone` rewrites `\documentclass[opts]{cls}` to
   `\documentclass[border=Xmm]{standalone}`, migrating only the options
   that compilation truly needs into `\usepackage{...}`. Known implicit
   dependencies are added (e.g. `amsart` ⇒ `amsmath`, `amssymb`).
6. The optimized file is **re-compiled before being written** so each
   output carries a `compilation_success` flag.

---

## Requirements

System packages:

- `latexmk`
- A LaTeX engine: `pdflatex` (preferred), `xelatex` and `lualatex`
  as fallbacks
- `poppler` (for `pdf2image`)
- `ghostscript` (used by `pdfCropMargins`; MacTeX bundles its own)

Python packages — install via:

```bash
pip install -r requirements.txt
```

This pulls in `pygments`, `pymupdf`, `pdf2image`, `pdfCropMargins` and
`Pillow`.

---

## Usage

### Command line

```bash
# single file
python import_optimizer.py --input my_figure.tex --output out_dir

# directory, parallel, with standalone conversion
python import_optimizer.py \
    --input examples/original \
    --output out_dir \
    --workers 8 --standalone --border 2mm
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input, -i` | Input `.tex` file or directory | required |
| `--output, -o` | Output directory | required |
| `--timeout, -t` | Per-compilation timeout (s) | `30` |
| `--workers, -w` | Parallel workers (use `1` for sequential) | `4` |
| `--max-files` | Limit how many files are processed | unlimited |
| `--standalone` | Rewrite `\documentclass` to `standalone` | off |
| `--border` | Border for the `standalone` class | `2mm` |
| `--quiet, -q` | Reduce log verbosity | off |

Directory inputs preserve their layout under the output directory.

### Programmatic API

```python
from import_optimizer import run

results = run(
    input_path="examples/original",
    output_dir="out_dir",
    timeout=30,
    workers=8,
    use_standalone=True,
    standalone_border="2mm",
)

for r in results:
    if r.status == "success":
        print(
            r.file,
            "removed:", r.removable_packages, r.removable_libraries,
            "compile ok:", r.compilation_success,
        )
```

`run(...)` returns a list of `OptimizationResult` dataclass instances:

```text
file                    str
status                  "success" | "error"
error                   Optional[str]
original_packages       list[str]
original_libraries      list[str]
removable_packages      list[str]
removable_libraries     list[str]
optimized_content       Optional[str]
output_file             Optional[str]
compilation_success     Optional[bool]
compilation_error       Optional[str]
```

### Custom compiler

If you need to swap in a different LaTeX backend (e.g. a remote
service, a Docker-based compiler, or one with custom flags), build your
own `LaTeXCompiler` and pass it to `TikZOptimizer` directly:

```python
from latex_compiler import LaTeXCompiler
from import_optimizer import TikZOptimizer

compiler = LaTeXCompiler(tex2img_fn=my_tex2img, timeout=60)
optimizer = TikZOptimizer(compiler=compiler)
result = optimizer.optimize_packages("figure.tex", use_standalone=True)
```

---

## Trying it out

Run the optimizer over the 20 bundled real-world samples (requires the
LaTeX toolchain — see *Requirements* above):

```bash
python import_optimizer.py \
    --input examples/original \
    --output examples/my_optimized \
    --workers 4 --timeout 30 --standalone
```

Reference numbers on a 2025 Apple Silicon laptop: 20/20 success,
≈ 50 s wall-clock, average dependency removal ≈ 78 %. The expected
output is shipped under `examples/optimized/` for direct comparison;
for instance:

```bash
diff examples/original/row_158190.tex \
     examples/optimized/row_158190.tex
```

shows that this file's preamble shrinks from 6 packages + 5 TikZ
libraries down to a single `\usepackage{tikz}` (plus a `standalone`
document class), losing 17 of 18 dependencies while still compiling.

---

## Limitations

- **Greedy ordering**: the result depends on declaration order; in rare
  cases an alternative subset would be smaller. Pass 2 mitigates this
  but does not guarantee a global optimum.
- **Compilation correctness only**: a package required for *visual*
  fidelity but not for compilation will be pruned. If visual identity
  matters, render the optimized file and diff against the original.
- **Recognised macros**: only `\usepackage{...}` (with optional
  `[...]`), `\usetikzlibrary{...}` and `\documentclass[...]{...}` are
  considered. Custom dependency macros are left untouched.
- **Implicit packages**: only `amsart ⇒ {amsmath, amssymb}` is
  hardcoded today. Add new mappings to `IMPLICIT_PACKAGES_MAP` in
  `import_optimizer.py` as needed.
