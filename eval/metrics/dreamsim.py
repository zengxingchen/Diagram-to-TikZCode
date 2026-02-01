from functools import cached_property
from typing import List, Union
from PIL import Image
import os

import torch
from torchmetrics import Metric

from dreamsim import dreamsim
from .utils import expand, load, infer_device


class DreamSim(Metric):
    """
    TorchMetrics-compatible perceptual similarity metric using DreamSim,
    now supporting batched evaluation for high parallelism.
    """

    higher_is_better = True

    def __init__(
        self,
        model_path: str = None,            # Optional local cache directory
        model_name: str = "ensemble",
        pretrained: bool = True,
        normalize: bool = True,
        preprocess: bool = True,                   # Whether to resize/center images before inference
        device: str = infer_device(),              # Automatically select CUDA or CPU
        dtype: torch.dtype = torch.float32,  # Use float32 to ensure compatibility with DreamSim
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self._device = device
        self.set_dtype(dtype)
        self.preprocess = preprocess
        # Resolve a usable cache dir: explicit arg > HF cache > torch hub default
        self.cache_dir = (
            model_path
            or os.environ.get("HF_HOME")
            or os.environ.get("TORCH_HOME")
            or os.path.expanduser("~/.cache/dreamsim")
        )

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__

    @cached_property
    def dreamsim(self):
        """
        Load the DreamSim model and processor from HuggingFace Hub.
        """
        model, processor = dreamsim(
            cache_dir=self.cache_dir,
            dreamsim_type=self.model_name,
            pretrained=self.pretrained,
            normalize_embeds=self.normalize,
            device=str(self.device)
        )

        # Move internal model components to correct dtype
        for extractor in model.extractor_list:
            extractor.model = extractor.model.to(self.dtype)
            extractor.proj = extractor.proj.to(self.dtype)

        return dict(
            model=model.to(self.dtype),
            processor=processor
        )

    @property
    def model(self):
        return self.dreamsim['model']

    @property
    def processor(self):
        return self.dreamsim['processor']

    @property
    def device(self):
        return self._device

    def _prepare_tensor(self, img: Union[Image.Image, str]) -> torch.Tensor:
        """Load, optional preprocess, and tensorize a single image."""
        pil = load(img)
        if self.preprocess:
            pil = expand(pil, max(pil.size), do_trim=True)
        t = self.processor(pil)  # expected shape [C,H,W] or [1,C,H,W]
        if isinstance(t, torch.Tensor):
            if t.ndim == 3:
                t = t.unsqueeze(0)
        else:
            # In case processor returns non-tensor, convert defensively
            t = torch.as_tensor(t)
            if t.ndim == 3:
                t = t.unsqueeze(0)
        return t  # [1,C,H,W]

    @torch.inference_mode()
    def update(self,
               img1: Union[Image.Image, str, List[Union[Image.Image, str]]],
               img2: Union[Image.Image, str, List[Union[Image.Image, str]]]):
        """
        Backward-compatible single-pair update.
        For batch use, prefer `update_batch` (returns per-pair scores).
        """
        if isinstance(img1, list) or isinstance(img2, list):
            # Delegate to batch version
            scores = self.update_batch(img1, img2)
            # (stats already updated inside update_batch)
            return scores

        t1 = self._prepare_tensor(img1)
        t2 = self._prepare_tensor(img2)

        t1 = t1.to(self.device, self.dtype, non_blocking=True)
        t2 = t2.to(self.device, self.dtype, non_blocking=True)

        with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu",
                            dtype=self.dtype, enabled=True):
            sim = self.model(t1, t2)  # expect shape [1] or scalar
        sim = sim.view(-1).to(dtype=torch.float64)
        score = (1.0 - sim)  # DSIM = 1 - similarity

        self.score += score.sum().detach().to(self.score.device)
        self.n_samples += torch.tensor(score.numel(), dtype=torch.long, device=self.score.device)
        # Return Python float for single case
        return score.item()

    @torch.inference_mode()
    def update_batch(self,
                     imgs1: List[Union[Image.Image, str]],
                     imgs2: List[Union[Image.Image, str]]) -> List[float]:
        """
        Batched update. Returns per-pair DSIM scores as Python floats.
        """
        assert isinstance(imgs1, list) and isinstance(imgs2, list), "update_batch expects lists"
        assert len(imgs1) == len(imgs2), "update_batch: imgs1 and imgs2 must have equal length"
        if len(imgs1) == 0:
            return []

        batch1 = [self._prepare_tensor(i) for i in imgs1]
        batch2 = [self._prepare_tensor(i) for i in imgs2]

        # Each element is [1,C,H,W]; concatenate along batch dim
        t1 = torch.cat(batch1, dim=0)
        t2 = torch.cat(batch2, dim=0)

        t1 = t1.to(self.device, self.dtype, non_blocking=True)
        t2 = t2.to(self.device, self.dtype, non_blocking=True)

        with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu",
                            dtype=self.dtype, enabled=True):
            sims = self.model(t1, t2)  # [B] or scalar
        sims = sims.view(-1).to(dtype=torch.float64)

        scores = (1.0 - sims)  # [B]
        self.score += scores.sum().detach().to(self.score.device)
        self.n_samples += torch.tensor(scores.numel(), dtype=torch.long, device=self.score.device)
        return scores.detach().cpu().tolist()

    def compute(self):
        return (self.score / self.n_samples).item()