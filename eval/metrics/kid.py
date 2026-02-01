from functools import cached_property
from typing import List, Union, Optional
import torch
from torch import nn
from torchmetrics.image.kid import KernelInceptionDistance as KID
from transformers import AutoImageProcessor, SiglipModel

from PIL import Image
from torch.cuda import is_available as is_cuda_available, is_bf16_supported

from .utils import expand, load, infer_device


class FeatureWrapper(nn.Module):
    """
    A wrapper around the SigLIP vision model to extract image features
    suitable for use in KID (Kernel Inception Distance) metric computation.
    """

    def __init__(self, model_path: str, device, dtype):
        super().__init__()
        self.model_path = model_path
        self._target_device = device
        self.dtype = dtype

    @cached_property
    def model(self):
        """
        Load the SigLIP vision model and move it to the target device with desired dtype.
        This uses local files only.
        """
        model = SiglipModel.from_pretrained(self.model_path, local_files_only=True)
        return model.to(self._target_device, dtype=self.dtype).eval()

    def forward(self, pixel_values):
        """
        Run inference on the model and return the pooled feature representation.
        """
        with torch.inference_mode():
            pixel_values = pixel_values.to(self._target_device, self.dtype)
            outputs = self.model.vision_model(pixel_values)
            # outputs = self.model(pixel_values)
            return outputs.pooler_output


class KernelInceptionDistance(KID):
    """
    Custom KID metric that uses SigLIP as the feature extractor instead of InceptionV3.
    """

    def __init__(
        self,
        model_path: str = "/data/szw/eval/siglip_local",  # Local path to the SigLIP model
        subset_size: int = 50,                            # KID subset size for polynomial MMD estimation
        preprocess: bool = True,                          # Whether to preprocess images before feeding them to the model
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        # Automatically determine device and dtype if not provided
        device = device or infer_device()
        dtype = dtype or (torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16)

        # Initialize the parent KID class with the SigLIP feature extractor
        super().__init__(
            subset_size=subset_size,
            feature=FeatureWrapper(
                model_path=model_path,
                device=device,
                dtype=dtype
            ),
            **kwargs
        )

        self.model_path = model_path
        self.preprocess = preprocess
        self._target_device = device

    @cached_property
    def processor(self):
        """
        Load the image processor used to preprocess images before inference.
        """
        return AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True)

    def open(self, img):
        """
        Load an image from file path or PIL.Image. Optionally preprocess it to square shape.
        """
        img = load(img)
        if self.preprocess:
            return expand(img, max(img.size), do_trim=True)
        return img

    def update(self, imgs: Union[Image.Image, str, List[Union[Image.Image, str]]], *args, **kwargs):
        """
        Preprocess images and feed them into the metric computation.
        Accepts a single image or a list of images (file path or PIL.Image).
        """
        if not isinstance(imgs, list):
            imgs = [imgs]

        # Preprocess and convert to pixel values using the HuggingFace processor
        processed = self.processor(
            [self.open(img) for img in imgs],
            return_tensors="pt"
        )["pixel_values"]

        # Convert to uint8 before updating metric buffer
        return super().update(processed, *args, **kwargs)

    def compute(self, *args, **kwargs):
        """
        Compute the final KID score from accumulated features.
        Returns the score as a tuple of float values.
        """
        return tuple(t.item() for t in super().compute(*args, **kwargs))