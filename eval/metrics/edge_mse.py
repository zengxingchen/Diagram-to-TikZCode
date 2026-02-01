from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class MSE:
    """
    Edge-aware z-score MSE score (score = 1 - MSE).

    Contract:
    - ``rendered_img`` must be grayscale float32 in ``[0, 1]`` (no
      automatic channel / range conversion is performed).
    - ``ground_truth_img`` is the dict returned by
      :py:meth:`wrap_ground_truth_img_from_path` ``{"bytes": ...}``.
    - Defaults: ``use_edge=True``; resize uses ``INTER_LINEAR``.
    - The raw score is clipped to ``[-1, 1]``; when ``as_percent=True``
      it is finally multiplied by 100.
    """

    def __init__(
        self,
        use_edge: bool = True,
        canny_thresh: Tuple[int, int] = (100, 200),
        dilate_ksize: int = 3,
        blur_ksize: int = 13,
        eps_std: float = 1e-6,
        clip_score: Tuple[float, float] = (-1.0, 1.0),
        resize_interpolation: int = cv2.INTER_LINEAR,
        as_percent: bool = False,
    ):
        self.use_edge = use_edge
        self.canny_thresh = canny_thresh
        self.dilate_ksize = dilate_ksize
        # GaussianBlur requires an odd kernel size.
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else max(blur_ksize - 1, 1)
        self.eps_std = eps_std
        self.clip_low, self.clip_high = clip_score
        self.resize_interpolation = resize_interpolation
        self.as_percent = as_percent

        # Running accumulator: per-image score then averaged on compute().
        self._sum_score: float = 0.0
        self._n: int = 0

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def reconstruction_reward(self, rendered_img: np.ndarray, ground_truth_img: Dict[str, Any]) -> Optional[float]:
        """
        Score a single sample:
        - (optional) Canny edges
        - z-score
        - l2 = mean((pred - gt) ** 2);   score = 1 - l2
        - clip to ``[-1, 1]``;  if ``as_percent=True`` multiply by 100.
        """
        try:
            # Strict input contract: grayscale float32 in [0, 1].
            self._assert_gray_float01(rendered_img)

            gt_img = self._load_ground_truth_img(ground_truth_img)

            # Resize prediction to match GT shape.
            if rendered_img.shape != gt_img.shape:
                pred_img = cv2.resize(rendered_img, (gt_img.shape[1], gt_img.shape[0]),
                                      interpolation=self.resize_interpolation)
            else:
                pred_img = rendered_img

            if self.use_edge:
                pred_img = self._canny_edge(pred_img)
                gt_img = self._canny_edge(gt_img)

            pred_img = self._normalize_img(pred_img)
            gt_img = self._normalize_img(gt_img)

            l2 = float(np.mean((pred_img - gt_img) ** 2))
            score = 1.0 - l2
            score = float(np.clip(score, self.clip_low, self.clip_high))
            if self.as_percent:
                score *= 100.0

            return score
        except Exception as e:
            print(f"Error in reconstruction_reward: {e}")
            return None

    def update(self, rendered_img: np.ndarray, ground_truth_img: Dict[str, Any]) -> None:
        """Score one image pair and accumulate into the running mean."""
        score = self.reconstruction_reward(rendered_img, ground_truth_img)
        if score is not None:
            self._sum_score += score
            self._n += 1

    def compute(self) -> Optional[float]:
        """Return the mean over accumulated samples, or None if empty."""
        if self._n == 0:
            return None
        return self._sum_score / self._n

    def reset(self) -> None:
        """Clear the running accumulator."""
        self._sum_score = 0.0
        self._n = 0

    # ---------------------------------------------------------------
    # Utilities (public)
    # ---------------------------------------------------------------
    @staticmethod
    def wrap_ground_truth_img_from_path(img_path: str) -> Dict[str, bytes]:
        """Read a GT image from disk and wrap it as ``{"bytes": ...}``."""
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Ground-truth image not found: {img_path}")
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        return {"bytes": img_bytes}

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------
    def _normalize_img(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        mean = float(img.mean())
        std = float(img.std())
        if std < self.eps_std:
            std = 1.0
        return (img - mean) / std

    def _canny_edge(self, img: np.ndarray) -> np.ndarray:
        """Canny -> (optional) dilation -> (optional) Gaussian blur; returns [0, 1] float."""
        ui8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        low, high = self.canny_thresh
        edges = cv2.Canny(ui8, low, high)

        if self.dilate_ksize and self.dilate_ksize > 1:
            kernel = np.ones((self.dilate_ksize, self.dilate_ksize), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        if self.blur_ksize and self.blur_ksize > 1:
            edges = cv2.GaussianBlur(edges, (self.blur_ksize, self.blur_ksize), 0)

        return edges.astype(np.float32) / 255.0

    @staticmethod
    def _assert_gray_float01(img: np.ndarray) -> None:
        """Assert the rendered image is 2-D grayscale float32 in [0, 1]."""
        if img is None:
            raise ValueError("Input image is None")
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a numpy ndarray")
        if img.ndim != 2:
            raise ValueError(f"Input image must be single-channel (grayscale), got shape {img.shape}")
        if img.dtype != np.float32:
            raise TypeError(f"Input image must be float32, got {img.dtype}")
        m, M = float(np.min(img)), float(np.max(img))
        if m < 0.0 - 1e-6 or M > 1.0 + 1e-6:
            raise ValueError(f"Input image values must be in [0, 1], got min={m}, max={M}")

    @staticmethod
    def _load_ground_truth_img(ground_truth_img: Dict[str, Any]) -> np.ndarray:
        """Decode the GT bytes into a grayscale float32 [0, 1] array."""
        img_bytes = ground_truth_img.get("bytes")
        if img_bytes is None:
            raise ValueError("Missing image bytes in ground_truth_img")
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        return img.astype(np.float32) / 255.0
