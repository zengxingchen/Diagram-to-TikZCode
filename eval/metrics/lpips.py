import os
import cv2
import numpy as np
import torch
import lpips
from typing import Dict, Any, Optional


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    - Compares two images in the feature space of a pretrained backbone
      (AlexNet / VGG).
    - Lower LPIPS means more similar images.
    - Value range: theoretically ``[0, +inf)``; for natural images
      typically in ``[0, 1]``.

    Input contract:
    - ``rendered_img`` must be grayscale float32 in ``[0, 1]``.
    - ``ground_truth_img`` is the dict returned by
      :py:meth:`wrap_ground_truth_img_from_path` ``{"bytes": ...}``.
    - Internally expanded to 3 channels and rescaled to ``[-1, 1]``.
    """

    def __init__(self, net: str = "vgg", resize_interpolation: int = cv2.INTER_LINEAR, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.resize_interpolation = resize_interpolation

        # Running accumulator.
        self._sum_score: float = 0.0
        self._n: int = 0

    def reconstruction_reward(self, rendered_img: np.ndarray, ground_truth_img: Dict[str, Any]) -> Optional[float]:
        try:
            self._assert_gray_float01(rendered_img)
            gt_img = self._load_ground_truth_img(ground_truth_img)

            # Resize prediction to match GT shape.
            if rendered_img.shape != gt_img.shape:
                pred_img = cv2.resize(rendered_img, (gt_img.shape[1], gt_img.shape[0]),
                                      interpolation=self.resize_interpolation)
            else:
                pred_img = rendered_img

            # Expand to 3 channels and rescale to [-1, 1] torch tensors.
            pred_tensor = self._to_tensor(pred_img).to(self.device)
            gt_tensor = self._to_tensor(gt_img).to(self.device)

            score = self.loss_fn(pred_tensor, gt_tensor).item()
            return float(score)
        except Exception as e:
            print(f"Error in LPIPS reconstruction_reward: {e}")
            return None

    def update(self, rendered_img: np.ndarray, ground_truth_img: Dict[str, Any]) -> None:
        score = self.reconstruction_reward(rendered_img, ground_truth_img)
        if score is not None:
            self._sum_score += score
            self._n += 1

    def compute(self) -> Optional[float]:
        if self._n == 0:
            return None
        return self._sum_score / self._n

    def reset(self) -> None:
        self._sum_score = 0.0
        self._n = 0

    @staticmethod
    def wrap_ground_truth_img_from_path(img_path: str) -> Dict[str, bytes]:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Ground-truth image not found: {img_path}")
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        return {"bytes": img_bytes}

    @staticmethod
    def _assert_gray_float01(img: np.ndarray) -> None:
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
            raise ValueError(f"Input image values must be in [0,1], got min={m}, max={M}")

    @staticmethod
    def _load_ground_truth_img(ground_truth_img: Dict[str, Any]) -> np.ndarray:
        img_bytes = ground_truth_img.get("bytes")
        if img_bytes is None:
            raise ValueError("Missing image bytes in ground_truth_img")
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        img3 = np.stack([img, img, img], axis=-1)  # HWC
        img3 = (img3 * 2.0 - 1.0).astype(np.float32)  # [0, 1] -> [-1, 1]
        tensor = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0)  # BCHW
        return tensor