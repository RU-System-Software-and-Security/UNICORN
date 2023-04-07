import warnings
from typing import Any, Dict, Union

import numpy as np
import PIL.Image
import torch

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform
from torchvision.transforms import functional as _F
from typing_extensions import Literal

from ._transform import _RandomApplyTransform
from ._utils import query_chw


DType = Union[torch.Tensor, PIL.Image.Image, features._Feature]


class ToTensor(Transform):
    _transformed_types = (PIL.Image.Image, np.ndarray)

    def __init__(self) -> None:
        warnings.warn(
            "The transform `ToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.Compose([transforms.ToImageTensor(), transforms.ConvertImageDtype()])`."
        )
        super().__init__()

    def _transform(self, inpt: Union[PIL.Image.Image, np.ndarray], params: Dict[str, Any]) -> torch.Tensor:
        return _F.to_tensor(inpt)


class Grayscale(Transform):
    _transformed_types = (features.Image, PIL.Image.Image, features.is_simple_tensor)

    def __init__(self, num_output_channels: Literal[1, 3] = 1) -> None:
        deprecation_msg = (
            f"The transform `Grayscale(num_output_channels={num_output_channels})` "
            f"is deprecated and will be removed in a future release."
        )
        if num_output_channels == 1:
            replacement_msg = (
                "transforms.ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY)"
            )
        else:
            replacement_msg = (
                "transforms.Compose(\n"
                "    transforms.ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY),\n"
                "    transforms.ConvertImageColorSpace(old_color_space=ColorSpace.GRAY, color_space=ColorSpace.RGB),\n"
                ")"
            )
        warnings.warn(f"{deprecation_msg} Instead, please use\n\n{replacement_msg}")

        super().__init__()
        self.num_output_channels = num_output_channels

    def _transform(self, inpt: DType, params: Dict[str, Any]) -> DType:
        output = _F.rgb_to_grayscale(inpt, num_output_channels=self.num_output_channels)
        if isinstance(inpt, features.Image):
            output = features.Image.new_like(inpt, output, color_space=features.ColorSpace.GRAY)
        return output


class RandomGrayscale(_RandomApplyTransform):
    _transformed_types = (features.Image, PIL.Image.Image, features.is_simple_tensor)

    def __init__(self, p: float = 0.1) -> None:
        warnings.warn(
            "The transform `RandomGrayscale(p=...)` is deprecated and will be removed in a future release. "
            "Instead, please use\n\n"
            "transforms.RandomApply(\n"
            "    transforms.Compose(\n"
            "        transforms.ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY),\n"
            "        transforms.ConvertImageColorSpace(old_color_space=ColorSpace.GRAY, color_space=ColorSpace.RGB),\n"
            "    )\n"
            "    p=...,\n"
            ")"
        )

        super().__init__(p=p)

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        num_input_channels, _, _ = query_chw(sample)
        return dict(num_input_channels=num_input_channels)

    def _transform(self, inpt: DType, params: Dict[str, Any]) -> DType:
        output = _F.rgb_to_grayscale(inpt, num_output_channels=params["num_input_channels"])
        if isinstance(inpt, features.Image):
            output = features.Image.new_like(inpt, output, color_space=features.ColorSpace.GRAY)
        return output
