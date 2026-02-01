"""
End-to-end evaluation for image-to-TikZ predictions.

Inputs
------
- ``--json``:    a JSON list of ``{predicted, ground_truth, ...}`` records
                 (typically produced by ``qwenvl_infer.py``)
- ``--pred-dir`` directory of rendered prediction PNGs (one per record)
- ``--ref-dir``  directory of ground-truth PNGs with matching filenames

Outputs
-------
- ``<output-dir>/metrics.csv``: per-image metric scores
- printed summary of aggregate scores

Each metric is opt-in via a ``--compute-*`` flag.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from metrics import (
    CrystalBLEUMetric, TextEditDistance,                 # text metrics
    DreamSimMetric, SigLIPSimilarity, KIDMetric,         # image-similarity metrics
    EdgeBasedMSE, PixelMSE, LPIPS, SSIM, PSNRMetric,     # reconstruction-quality metrics
)


# ---------------------------------------------------------------------------
# Shared image-pair iteration
# ---------------------------------------------------------------------------

def _png_list(directory: str) -> List[str]:
    return sorted(f for f in os.listdir(directory) if f.endswith(".png"))


def _load_pair(pred_path: str, ref_path: str, mode: str = "RGB"
               ) -> Tuple[Image.Image, Image.Image]:
    return (Image.open(pred_path).convert(mode),
            Image.open(ref_path).convert(mode))


def _iterate_pairs(
    pred_dir: str, ref_dir: str,
    process_func: Callable[[str, str, str], Optional[Dict[str, Any]]],
    desc: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Iterate matching PNGs under ``pred_dir`` and ``ref_dir`` and apply
    ``process_func(pred_path, ref_path, fname)`` to each pair."""
    records: List[Dict[str, Any]] = []
    for fname in tqdm(_png_list(pred_dir), desc=desc):
        pred_path = os.path.join(pred_dir, fname)
        ref_path = os.path.join(ref_dir, fname)
        if not os.path.exists(ref_path):
            continue
        try:
            result = process_func(pred_path, ref_path, fname)
            if result is not None:
                records.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: failed on {fname}: {exc}")
    return records, len(records)


def _load_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data or not isinstance(data, list):
        raise ValueError("JSON data is empty or not a list")
    if not all("ground_truth" in item and "predicted" in item for item in data):
        raise ValueError("JSON missing required fields: 'ground_truth' / 'predicted'")
    return data


# ---------------------------------------------------------------------------
# Per-metric evaluators
# ---------------------------------------------------------------------------

def evaluate_dsim(pred_dir: str, ref_dir: str):
    # `DreamSimMetric` is a torchmetrics.Metric: update() accumulates state
    # and compute() returns the running mean. To get a true per-image
    # score we reset() before every sample; the underlying model is held
    # in a cached_property, so reset() is essentially free.
    metric = DreamSimMetric()

    def step(pred_path, ref_path, fname):
        pred_img, ref_img = _load_pair(pred_path, ref_path, "RGB")
        metric.reset()
        metric.update(pred_img, ref_img)
        return {"filename": fname, "DSIM": float(metric.compute())}

    return _iterate_pairs(pred_dir, ref_dir, step, "Computing DSIM")


def evaluate_siglip(pred_dir: str, ref_dir: str, model_name: str):
    metric = SigLIPSimilarity(model_path=model_name, mode="cos")

    def step(pred_path, ref_path, fname):
        pred_img, ref_img = _load_pair(pred_path, ref_path, "RGB")
        metric.reset()
        metric.update(pred_img, ref_img)
        return {"filename": fname, "siglip": float(metric.compute())}

    return _iterate_pairs(pred_dir, ref_dir, step, "Computing siglip")


def evaluate_structural_sim(pred_dir: str, ref_dir: str):
    metric = SSIM(data_range=1.0, resize_to_match=True, preprocess=True)

    def step(pred_path, ref_path, fname):
        pred_img, ref_img = _load_pair(pred_path, ref_path, "RGB")
        metric.reset()
        metric.update(pred_img, ref_img)
        return {"filename": fname, "StructuralSIM": float(metric.compute())}

    return _iterate_pairs(pred_dir, ref_dir, step, "Computing StructuralSIM")


def _evaluate_recon_metric(
    pred_dir: str, ref_dir: str, meter, key: str, desc: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Common loop for reconstruction-quality metrics whose API is
    ``meter.reconstruction_reward(pred_gray_float, gt_dict)``."""
    def step(pred_path, ref_path, fname):
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            return None
        pred_img = pred_img.astype(np.float32) / 255.0
        gt_dict = meter.wrap_ground_truth_img_from_path(ref_path)
        score = meter.reconstruction_reward(pred_img, gt_dict)
        if score is None:
            return None
        return {"filename": fname, key: float(score)}
    return _iterate_pairs(pred_dir, ref_dir, step, desc)


def evaluate_mse(pred_dir: str, ref_dir: str):
    meter = EdgeBasedMSE(
        use_edge=True, canny_thresh=(100, 200),
        dilate_ksize=3, blur_ksize=13,
        resize_interpolation=cv2.INTER_LINEAR, as_percent=True,
    )
    return _evaluate_recon_metric(pred_dir, ref_dir, meter, "MSE", "Computing MSE (edge)")


def evaluate_mse_nocanny(pred_dir: str, ref_dir: str):
    meter = PixelMSE(resize_interpolation=cv2.INTER_LINEAR, as_percent=True)
    return _evaluate_recon_metric(
        pred_dir, ref_dir, meter, "MSE_nocanny", "Computing MSE (no Canny)",
    )


def evaluate_lpips(pred_dir: str, ref_dir: str,
                   net: str = "vgg", device: str = "cuda:0"):
    meter = LPIPS(net=net, device=device)
    return _evaluate_recon_metric(pred_dir, ref_dir, meter, "LPIPS", "Computing LPIPS")


def evaluate_psnr(pred_dir: str, ref_dir: str):
    meter = PSNRMetric(resize_interpolation=cv2.INTER_LINEAR)
    return _evaluate_recon_metric(pred_dir, ref_dir, meter, "PSNR", "Computing PSNR")


def evaluate_kid(pred_dir: str, ref_dir: str, model_path: str):
    metric = KIDMetric(model_path=model_path, subset_size=50)

    def step(pred_path, ref_path, fname):
        pred_img, ref_img = _load_pair(pred_path, ref_path, "RGB")
        metric.update(ref_img, real=True)
        metric.update(pred_img, real=False)
        return None

    _, total = _iterate_pairs(pred_dir, ref_dir, step, "Computing KID")
    kid_score, _ = metric.compute()
    return [{"filename": "global", "KID": kid_score}], total


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _merge_by_filename(*metric_results: List[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for records in metric_results:
        for item in records:
            fname = item.get("filename", "global")
            merged[fname].update(item)
            merged[fname]["filename"] = fname
    return list(merged.values())


def evaluate_all_metrics(
    json_path: str, pred_dir: str, ref_dir: str, output_dir: str,
    compute_cbleu: bool = False, compute_ted: bool = False,
    compute_dsim: bool = False, compute_siglip: bool = False,
    compute_kid: bool = False, compute_mse: bool = False,
    compute_mse_nocanny: bool = False, compute_ssim: bool = False,
    compute_lpips: bool = False, compute_psnr: bool = False,
    siglip_model: str = "google/siglip-so400m-patch14-384",
    lpips_device: str = "cuda:0",
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    per_image: List[List[Dict[str, Any]]] = []

    # --- Image-based metrics ----------------------------------------------
    if compute_dsim:
        recs, _ = evaluate_dsim(pred_dir, ref_dir)
        per_image.append(recs)
        if recs:
            results["DSIM"] = pd.DataFrame(recs)["DSIM"].mean() * 100

    if compute_siglip:
        recs, _ = evaluate_siglip(pred_dir, ref_dir, model_name=siglip_model)
        per_image.append(recs)
        if recs:
            results["siglip"] = pd.DataFrame(recs)["siglip"].mean() * 100

    if compute_mse:
        recs, _ = evaluate_mse(pred_dir, ref_dir)
        per_image.append(recs)
        if recs:
            results["MSE"] = pd.DataFrame(recs)["MSE"].mean()        # already percent

    if compute_mse_nocanny:
        recs, _ = evaluate_mse_nocanny(pred_dir, ref_dir)
        per_image.append(recs)
        if recs:
            results["MSE_nocanny"] = pd.DataFrame(recs)["MSE_nocanny"].mean()

    if compute_lpips:
        recs, _ = evaluate_lpips(pred_dir, ref_dir, net="vgg", device=lpips_device)
        per_image.append(recs)
        if recs:
            results["LPIPS"] = pd.DataFrame(recs)["LPIPS"].mean()

    if compute_ssim:
        recs, _ = evaluate_structural_sim(pred_dir, ref_dir)
        per_image.append(recs)
        if recs:
            results["StructuralSIM"] = pd.DataFrame(recs)["StructuralSIM"].mean()

    if compute_psnr:
        recs, _ = evaluate_psnr(pred_dir, ref_dir)
        per_image.append(recs)
        if recs:
            results["PSNR"] = pd.DataFrame(recs)["PSNR"].mean()      # dB

    if compute_kid:
        recs, _ = evaluate_kid(pred_dir, ref_dir, model_path=siglip_model)
        per_image.append(recs)
        if recs:
            results["KID"] = recs[0]["KID"]

    # --- Compilation coverage --------------------------------------------
    data = _load_json(json_path)
    compiled = len(_png_list(pred_dir))
    total = len(data)
    results["compiled"] = (
        f"{compiled} / {total} ({compiled / max(total, 1) * 100:.1f}%)"
    )

    # Write the per-image CSV
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(_merge_by_filename(*per_image)).to_csv(
        os.path.join(output_dir, "metrics.csv"), index=False,
    )

    # --- Text metrics -----------------------------------------------------
    if compute_cbleu:
        print("\nComputing CrystalBLEU...")
        refs = [[item["ground_truth"].strip()] for item in data]
        preds = [item["predicted"].strip() for item in data]
        bleu = CrystalBLEUMetric(corpus=[r[0] for r in refs], use_cache=False)
        bleu.update(refs, preds)
        results["CrystalBLEU"] = bleu.compute() * 100

    if compute_ted:
        print("\nComputing TextEditDistance...")
        refs = [item["ground_truth"].strip() for item in data]
        preds = [item["predicted"].strip() for item in data]
        ted = TextEditDistance(language="en")
        ted.update(preds, refs)
        results["TextEditDistance"] = ted.compute() * 100

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate image-to-TikZ predictions on a battery of "
                    "text, image-similarity and reconstruction metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json", required=True,
                   help="Prediction JSON (from qwenvl_infer.py).")
    p.add_argument("--pred-dir", required=True,
                   help="Directory of rendered prediction PNGs.")
    p.add_argument("--ref-dir", required=True,
                   help="Directory of reference (ground-truth) PNGs.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save per-image metrics.csv.")
    p.add_argument("--siglip-model",
                   default="google/siglip-so400m-patch14-384",
                   help="HuggingFace model id (or local dir) for SigLIP / KID.")
    p.add_argument("--lpips-device", default="cuda:0",
                   help="Device for the LPIPS backbone.")
    # Metric switches
    p.add_argument("--compute-cbleu",       action="store_true")
    p.add_argument("--compute-ted",         action="store_true")
    p.add_argument("--compute-dsim",        action="store_true")
    p.add_argument("--compute-siglip",      action="store_true")
    p.add_argument("--compute-kid",         action="store_true")
    p.add_argument("--compute-mse",         action="store_true",
                   help="Edge-based pixel MSE (uses Canny).")
    p.add_argument("--compute-mse-nocanny", action="store_true",
                   help="Plain pixel MSE without Canny.")
    p.add_argument("--compute-ssim",        action="store_true",
                   help="Structural similarity (SSIM).")
    p.add_argument("--compute-lpips",       action="store_true")
    p.add_argument("--compute-psnr",        action="store_true")
    return p


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = _build_arg_parser().parse_args()

    results = evaluate_all_metrics(
        json_path=args.json,
        pred_dir=args.pred_dir,
        ref_dir=args.ref_dir,
        output_dir=args.output_dir,
        compute_cbleu=args.compute_cbleu,
        compute_ted=args.compute_ted,
        compute_dsim=args.compute_dsim,
        compute_siglip=args.compute_siglip,
        compute_kid=args.compute_kid,
        compute_mse=args.compute_mse,
        compute_mse_nocanny=args.compute_mse_nocanny,
        compute_ssim=args.compute_ssim,
        compute_lpips=args.compute_lpips,
        compute_psnr=args.compute_psnr,
        siglip_model=args.siglip_model,
        lpips_device=args.lpips_device,
    )

    print("\nOverall results:")
    for key, val in results.items():
        if isinstance(val, str):
            print(f"  {key:18s} {val}")
        else:
            print(f"  {key:18s} {val:.4f}")
