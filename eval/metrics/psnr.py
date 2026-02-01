import os
import cv2
import numpy as np
from typing import Dict, Any, Optional
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class PSNR:
    """
    Peak Signal-to-Noise Ratio (PSNR).

    - Measures pixel-domain similarity; higher PSNR means closer images.
    - Typical range ~20-50 dB depending on the task; +inf for a perfect
      reconstruction.

    Input contract (kept consistent with the LPIPS metric):
    - ``rendered_img`` must be grayscale float32 in ``[0, 1]``.
    - ``ground_truth_img`` is the dict returned by
      :py:meth:`wrap_ground_truth_img_from_path` ``{"bytes": ...}``.
    - The GT is decoded to grayscale and normalized to ``[0, 1]``.
    - If shapes differ, the prediction is resized to the GT shape using
      ``resize_interpolation``.
    """

    def __init__(self, resize_interpolation: int = cv2.INTER_LINEAR):
        self.resize_interpolation = resize_interpolation

        # Running accumulator.
        self._sum_score: float = 0.0
        self._n: int = 0

    def reconstruction_reward(self, rendered_img: np.ndarray, ground_truth_img: Dict[str, Any]) -> Optional[float]:
        """Compute PSNR (dB) for one image pair, or None on failure."""
        try:
            self._assert_gray_float01(rendered_img)
            gt_img = self._load_ground_truth_img(ground_truth_img)

            # Resize prediction to match GT shape.
            if rendered_img.shape != gt_img.shape:
                pred_img = cv2.resize(
                    rendered_img,
                    (gt_img.shape[1], gt_img.shape[0]),
                    interpolation=self.resize_interpolation
                )
            else:
                pred_img = rendered_img

            # skimage PSNR: data_range=1.0 because inputs are in [0, 1].
            psnr = compare_psnr(gt_img, pred_img, data_range=1.0)
            return float(psnr)
        except Exception as e:
            print(f"Error in PSNR reconstruction_reward: {e}")
            return None

    def update(self, rendered_img: np.ndarray, ground_truth_img: Dict[str, Any]) -> None:
        """Score one pair and accumulate; failed samples are skipped."""
        score = self.reconstruction_reward(rendered_img, ground_truth_img)
        if score is not None:
            self._sum_score += score
            self._n += 1

    def compute(self) -> Optional[float]:
        """Return the arithmetic mean PSNR (dB), or None if no samples."""
        if self._n == 0:
            return None
        return self._sum_score / self._n

    def reset(self) -> None:
        """Clear the running accumulator."""
        self._sum_score = 0.0
        self._n = 0

    @staticmethod
    def wrap_ground_truth_img_from_path(img_path: str) -> Dict[str, bytes]:
        """Read a GT image from disk and wrap it as ``{"bytes": ...}``."""
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Ground-truth image not found: {img_path}")
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        return {"bytes": img_bytes}

    @staticmethod
    def _assert_gray_float01(img: np.ndarray) -> None:
        """Assert input is 2-D grayscale float32 in [0, 1]."""
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
