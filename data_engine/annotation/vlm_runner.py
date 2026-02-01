"""
Shared vLLM helper for batched VLM annotation jobs.

- Lazy ``LLM`` / ``processor`` initialisation (so ``--help`` doesn't try
  to spin up a model).
- ``normalize_image_item`` / ``load_paths_from_parquet_parallel``:
  accept image columns that contain raw bytes, ``{path,bytes}`` dicts,
  ``data:image/...`` URIs, or local paths; cache bytes-blobs to a
  temporary directory and verify each PNG.
- ``run_batched``: a generic main-loop that takes an input parquet, a
  per-row prompt builder, batch size, and writes one JSONL line per
  image with its raw text reply (with simple resume support).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from PIL import Image


# ===========================================================================
# Image normalisation
# ===========================================================================

_DEFAULT_TMP_DIR: Optional[str] = None
_BYTES_PATH_CACHE: Dict[str, str] = {}


def get_tmp_dir() -> str:
    """Return (and lazily create) the directory used to spill image bytes."""
    global _DEFAULT_TMP_DIR
    if _DEFAULT_TMP_DIR is None:
        _DEFAULT_TMP_DIR = os.environ.get(
            "VLM_TMP_DIR", tempfile.mkdtemp(prefix="vlm_imgs_")
        )
        os.makedirs(_DEFAULT_TMP_DIR, exist_ok=True)
    return _DEFAULT_TMP_DIR


def _bytes_to_tmp_path(b: bytes, idx_hint: int, ext: str = ".png") -> str:
    h = hashlib.sha1(b).hexdigest()
    cached = _BYTES_PATH_CACHE.get(h)
    if cached and os.path.exists(cached):
        return cached
    p = Path(get_tmp_dir()) / f"img_{idx_hint:08d}_{h[:8]}{ext}"
    if not p.exists():
        with open(p, "wb") as f:
            f.write(b)
    _BYTES_PATH_CACHE[h] = str(p)
    return str(p)


def normalize_image_item(
    x: Any, idx: int, verify: bool = True,
) -> Optional[str]:
    """Coerce one parquet image cell into a local file path (or URL).

    Returns ``None`` if the cell is unusable.
    Returns ``"__INVALID__:<path>"`` if a path was produced but the file
    cannot be opened by Pillow (caller decides how to handle).
    """
    if isinstance(x, (bytes, bytearray, memoryview)):
        path = _bytes_to_tmp_path(bytes(x), idx)
    elif isinstance(x, dict):
        if isinstance(x.get("path"), str) and x["path"]:
            path = x["path"]
        elif "bytes" in x and isinstance(x["bytes"], (bytes, bytearray, memoryview)):
            path = _bytes_to_tmp_path(bytes(x["bytes"]), idx)
        else:
            return None
    elif isinstance(x, str) and x.startswith("data:image/"):
        header, b64 = x.split(",", 1)
        ext = ".png"
        if "jpeg" in header or "jpg" in header:
            ext = ".jpg"
        elif "webp" in header:
            ext = ".webp"
        path = _bytes_to_tmp_path(base64.b64decode(b64), idx, ext)
    elif isinstance(x, str):
        path = x
    else:
        return None

    if not verify or path.startswith("http"):
        return path
    try:
        with Image.open(path) as im:
            im.verify()
        return path
    except Exception:
        return f"__INVALID__:{path}"


def load_paths_from_parquet_parallel(
    parquet_path: str,
    col: str = "image",
    limit: Optional[int] = None,
    max_workers: int = 8,
    drop_invalid: bool = False,
) -> List[str]:
    """Load and normalise an image column to a list of file paths.

    Falls back to common alternative column names if ``col`` is missing.
    Optional ``drop_invalid=True`` filters out items that could not be
    verified by Pillow (otherwise they are kept as ``__INVALID__:...``).
    """
    df = pd.read_parquet(parquet_path)
    if col not in df.columns:
        for c in ("image", "path", "image_path", "file", "file_path",
                  "uri", "url"):
            if c in df.columns:
                col = c
                break
    series = df[col].iloc[:limit] if limit else df[col]
    items = series.tolist()

    results: List[Optional[str]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(normalize_image_item, v, i): i
            for i, v in enumerate(items)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = None

    if drop_invalid:
        return [
            p for p in results
            if p and not p.startswith("__INVALID__:")
        ]
    return [p for p in results if p]


# ===========================================================================
# vLLM lazy loader
# ===========================================================================

class VLMBackend:
    """Lazy holder for a vLLM model + its processor."""

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 16384,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.limit_mm_per_prompt = limit_mm_per_prompt or {
            "image": 10, "video": 10,
        }
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._llm = None
        self._processor = None
        self._sampling_params = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        print(f"Loading vLLM model: {self.model_path}")
        self._llm = LLM(
            model=self.model_path,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._sampling_params = SamplingParams(
            temperature=self.temperature, top_p=1.0,
            repetition_penalty=1.0, max_tokens=self.max_tokens,
            stop_token_ids=[],
        )

    @property
    def llm(self):
        self._ensure_loaded()
        return self._llm

    @property
    def processor(self):
        self._ensure_loaded()
        return self._processor

    @property
    def sampling_params(self):
        self._ensure_loaded()
        return self._sampling_params

    def build_llm_inputs(
        self, image_path: str, system_prompt: str, user_text: str,
    ) -> Dict[str, Any]:
        """Build the multi-modal inputs dict for ``llm.generate``."""
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image_path,
                 "min_pixels": 224 * 224,
                 "max_pixels": 1280 * 28 * 28},
                {"type": "text", "text": user_text},
            ]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True,
        )
        mm_data: Dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        return {
            "prompt": prompt, "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }


# ===========================================================================
# Batched main loop
# ===========================================================================

def run_batched(
    parquet_path: str,
    save_jsonl: str,
    backend: VLMBackend,
    inputs_builder: Callable[[VLMBackend, str], Dict[str, Any]],
    image_col: str = "image",
    batch_size: int = 8,
    limit_first_n: Optional[int] = None,
    normalize_workers: int = 8,
    resume: bool = True,
    record_index: bool = True,
) -> int:
    """Generic batched VLM annotation loop.

    Each row in the input parquet's ``image_col`` is normalised to a
    local file path and fed to ``inputs_builder`` (which receives the
    backend and the path, and returns the ``llm.generate`` inputs dict).
    The model's raw text reply is appended to ``save_jsonl`` as one
    JSON record per row::

        {"index": <row idx>, "path": <image path>, "raw": <model text>}

    Returns the number of records written.
    """
    save_dir = os.path.dirname(save_jsonl) or "."
    os.makedirs(save_dir, exist_ok=True)

    processed = 0
    if resume and os.path.exists(save_jsonl):
        with open(save_jsonl, "r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)
        print(f"[resume] {processed} records already present; resuming")
        fout = open(save_jsonl, "a", encoding="utf-8")
    else:
        fout = open(save_jsonl, "w", encoding="utf-8")

    img_paths = load_paths_from_parquet_parallel(
        parquet_path, col=image_col, limit=limit_first_n,
        max_workers=normalize_workers,
    )
    total = len(img_paths)
    print(f"Loaded {total} images from {parquet_path}")

    written = 0
    for i in range(processed, total, batch_size):
        batch = img_paths[i:i + batch_size]
        valid_inputs = []
        valid_idx_in_batch: List[int] = []
        for j, path in enumerate(batch):
            if path and not path.startswith("__INVALID__:"):
                valid_inputs.append(inputs_builder(backend, path))
                valid_idx_in_batch.append(j)

        if valid_inputs:
            outs = backend.llm.generate(
                valid_inputs, sampling_params=backend.sampling_params,
            )
        else:
            outs = []

        out_iter = iter(outs)
        records = []
        for j, path in enumerate(batch):
            rec: Dict[str, Any] = {}
            if record_index:
                rec["index"] = i + j
            rec["path"] = path
            if not path or path.startswith("__INVALID__:"):
                rec["raw"] = "INVALID IMAGE"
            else:
                o = next(out_iter)
                rec["raw"] = (o.outputs[0].text or "").strip()
            records.append(json.dumps(rec, ensure_ascii=False))

        fout.write("\n".join(records) + "\n")
        fout.flush()
        written += len(records)
        print(f"  wrote {i + len(batch)}/{total}")

    fout.close()
    print(f"Done. JSONL -> {save_jsonl}")
    return written
