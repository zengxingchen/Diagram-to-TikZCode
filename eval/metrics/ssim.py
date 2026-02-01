from __future__ import annotations

from typing import List, Union, Tuple
from PIL import Image

import numpy as np
import torch
from torchmetrics import Metric
from torch.cuda import is_available as is_cuda_available, is_bf16_supported

# Use only torchmetrics' SSIM implementation.
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as ssim_fn_tm
)

from .utils import expand, load, infer_device


class StructuralSIM(Metric):
    """
    Structural Similarity Index (SSIM) backed by torchmetrics.

    - ``higher_is_better = True``
    - Theoretical range ``[-1, 1]``; in practice typically ``[0, 1]``;
      ``1`` means identical.
    - Defaults follow the common literature setting:
      ``kernel_size=11, sigma=1.5, k1=0.01, k2=0.03``.
    - Accepts PIL images or file paths; can optionally pad images to a
      square and resize prediction to match the GT shape.
    """

    higher_is_better = True

    def __init__(
        self,
        data_range: float = 1.0,                 # Range of input tensors / arrays (default [0, 1]).
        kernel_size: int = 11,                   # SSIM window size (11 is standard).
        sigma: float = 1.5,                      # Std-dev of the Gaussian window.
        k1: float = 0.01,
        k2: float = 0.03,
        resize_to_match: bool = True,            # Auto-resize pred to the GT shape.
        preprocess: bool = True,                 # Pad to a square via util.image.expand (removes black borders).
        device: str = infer_device(),
        dtype: torch.dtype = torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_range = float(data_range)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.resize_to_match = bool(resize_to_match)
        self.preprocess = bool(preprocess)

        self._device = device
        self.set_dtype(dtype)

        # Aggregation state.
        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    # ---------- Utilities ----------
    @staticmethod
    def _pil_resize_to(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        # Mirror OpenCV's INTER_LINEAR with PIL's BILINEAR.
        return img.resize(size, resample=Image.BILINEAR)

    @staticmethod
    def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
        """
        PIL -> torch.Tensor (1, C, H, W), float32 in [0, 1].
        Grayscale is expanded to 3 channels to match common SSIM settings.
        """
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        if img.mode == "L":
            img = img.convert("RGB")

        arr = np.asarray(img).astype(np.float32) / 255.0      # H W C, [0, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1 C H W
        return t.to(torch.float32)

    # ---------- Metric API ----------
    def update(
        self,
        img1: Union[Image.Image, str, List[Union[Image.Image, str]]],
        img2: Union[Image.Image, str, List[Union[Image.Image, str]]],
    ):
        """
        Inputs can be a single image (PIL or path) or a list (with equal length).
        """
        if isinstance(img1, list) or isinstance(img2, list):
            assert type(img1) == type(img2) and len(img1) == len(img2)
        else:
            img1, img2 = [img1], [img2]

        for i1, i2 in zip(img1, img2):
            # Load -> PIL.Image
            p1 = load(i1)
            p2 = load(i2)

            # Optional square padding preprocess.
            if self.preprocess:
                p1 = expand(p1, max(p1.size), do_trim=True)
                p2 = expand(p2, max(p2.size), do_trim=True)

            # Resize prediction to match GT shape.
            if self.resize_to_match and p1.size != p2.size:
                p1 = self._pil_resize_to(p1, p2.size)

            # To tensor on device; torchmetrics expects (N, C, H, W).
            t1 = self._pil_to_tensor01(p1).to(self.device)
            t2 = self._pil_to_tensor01(p2).to(self.device)

            with torch.inference_mode():
                ssim_val = ssim_fn_tm(
                    t1, t2,
                    data_range=self.data_range,
                    kernel_size=self.kernel_size,
                    sigma=self.sigma,
                    k1=self.k1, k2=self.k2,
                    reduction="elementwise_mean",
                )

            self.score += float(ssim_val.detach().cpu().item())
            self.n_samples += 1

    def compute(self) -> float:
        if int(self.n_samples.item()) == 0:
            return float("nan")
        return (self.score / self.n_samples).item()
