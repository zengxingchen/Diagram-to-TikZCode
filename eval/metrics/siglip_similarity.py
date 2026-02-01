from functools import cached_property
from typing import List, Optional, Literal

from PIL import Image
import torch
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import AutoImageProcessor, SiglipVisionModel

from .utils import expand, load, infer_device


class ImageSim(Metric):
    """
    Image similarity metric based on the SigLIP model (online).
    Computes cosine similarity between vision embeddings of input images.
    """

    higher_is_better = True

    def __init__(
        self,
        model_path: str = "google/siglip-so400m-patch14-384",  # HuggingFace model id
        mode: Literal["cos", "cos_avg"] = "cos",
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.preprocess = preprocess
        self.mode = mode
        self._device = device
        self.set_dtype(dtype)

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__ + f" ({self.mode.upper().replace('_', '-')})"

    @cached_property
    def model(self):
        """Auto-download SigLIP from HuggingFace (GPU + bfloat16 supported)."""
        return SiglipVisionModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype
        ).to(self.device)

    @cached_property
    def processor(self):
        """Auto-load the image processor."""
        return AutoImageProcessor.from_pretrained(self.model_path)

    def get_vision_features(self, image: Optional[Image.Image | str] = None, text: Optional[str] = None):
        if image is not None:
            image = load(image)
            if self.preprocess:
                image = expand(image, max(image.size), do_trim=True)

        with torch.inference_mode():
            if text is not None:
                encoding = self.processor(text=text, images=image, return_tensors="pt").to(self.device, self.dtype)
            else:
                encoding = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)

            if self.mode == "cos":
                return self.model(**encoding).pooler_output.squeeze()
            elif self.mode == "cos_avg":
                return self.model(**encoding).last_hidden_state.squeeze().mean(dim=0)
            else:
                return self.model(**encoding).last_hidden_state.squeeze()

    def get_similarity(
        self,
        img1: Optional[Image.Image | str] = None,
        img2: Optional[Image.Image | str] = None,
        text1: Optional[str] = None,
        text2: Optional[str] = None,
    ):
        img1_feats = self.get_vision_features(img1, text1)
        img2_feats = self.get_vision_features(img2, text2)

        if img1_feats.is_mps:
            img1_feats, img2_feats = img1_feats.cpu(), img2_feats.cpu()

        if img1_feats.ndim > 1:
            dists = 1 - pairwise_cosine_similarity(img1_feats.double(), img2_feats.double()).cpu().numpy()
            return 2 * tanh(-emd2(M=dists, a=list(), b=list())) + 1
        else:
            return F.cosine_similarity(img1_feats.double(), img2_feats.double(), dim=0).item()

    def update(
        self,
        img1: Optional[Image.Image | str | List[Image.Image | str]] = None,
        img2: Optional[Image.Image | str | List[Image.Image | str]] = None,
        text1: Optional[str | List[str]] = None,
        text2: Optional[str | List[str]] = None,
    ):
        inputs = dict()
        for key, value in dict(img1=img1, img2=img2, text1=text1, text2=text2).items():
            if value is not None:
                inputs[key] = value if isinstance(value, list) else [value]

        assert not ({"img1", "text1"}.isdisjoint(inputs.keys()) or {"img2", "text2"}.isdisjoint(inputs.keys()))
        assert len(set(map(len, inputs.values()))) == 1

        for inpt in zip(*inputs.values()):
            self.score += self.get_similarity(**dict(zip(inputs.keys(), inpt)))
            self.n_samples += 1

    def compute(self):
        return (self.score / self.n_samples).item()