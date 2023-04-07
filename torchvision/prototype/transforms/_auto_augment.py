import math
import numbers
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import PIL.Image
import torch

from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.prototype import features
from torchvision.prototype.transforms import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.prototype.transforms.functional._meta import get_chw

from ._utils import _isinstance

K = TypeVar("K")
V = TypeVar("V")


class _AutoAugmentBase(Transform):
    def __init__(
        self,
        *,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
    ) -> None:
        super().__init__()
        self.interpolation = interpolation

        if not isinstance(fill, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate fill arg")
        self.fill = fill

    def _get_random_item(self, dct: Dict[K, V]) -> Tuple[K, V]:
        keys = tuple(dct.keys())
        key = keys[int(torch.randint(len(keys), ()))]
        return key, dct[key]

    def _extract_image(
        self,
        sample: Any,
        unsupported_types: Tuple[Type, ...] = (features.BoundingBox, features.SegmentationMask),
    ) -> Tuple[int, Union[PIL.Image.Image, torch.Tensor, features.Image]]:
        sample_flat, _ = tree_flatten(sample)
        images = []
        for id, inpt in enumerate(sample_flat):
            if _isinstance(inpt, (features.Image, PIL.Image.Image, features.is_simple_tensor)):
                images.append((id, inpt))
            elif isinstance(inpt, unsupported_types):
                raise TypeError(f"Inputs of type {type(inpt).__name__} are not supported by {type(self).__name__}()")

        if not images:
            raise TypeError("Found no image in the sample.")
        if len(images) > 1:
            raise TypeError(
                f"Auto augment transformations are only properly defined for a single image, but found {len(images)}."
            )
        return images[0]

    def _put_into_sample(self, sample: Any, id: int, item: Any) -> Any:
        sample_flat, spec = tree_flatten(sample)
        sample_flat[id] = item
        return tree_unflatten(sample_flat, spec)

    def _apply_image_transform(
        self,
        image: Any,
        transform_id: str,
        magnitude: float,
        interpolation: InterpolationMode,
        fill: Union[int, float, Sequence[int], Sequence[float]],
    ) -> Any:

        # Fill = 0 is not equivalent to None, https://github.com/pytorch/vision/issues/6517
        # So, we have to put fill as None if fill == 0
        fill_: Optional[Union[int, float, Sequence[int], Sequence[float]]]
        if isinstance(fill, int) and fill == 0:
            fill_ = None
        else:
            fill_ = fill

        if transform_id == "Identity":
            return image
        elif transform_id == "ShearX":
            # magnitude should be arctan(magnitude)
            # official autoaug: (1, level, 0, 0, 1, 0)
            # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
            # compared to
            # torchvision:      (1, tan(level), 0, 0, 1, 0)
            # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
            return F.affine(
                image,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(magnitude)), 0.0],
                interpolation=interpolation,
                fill=fill_,
                center=[0, 0],
            )
        elif transform_id == "ShearY":
            # magnitude should be arctan(magnitude)
            # See above
            return F.affine(
                image,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(magnitude))],
                interpolation=interpolation,
                fill=fill_,
                center=[0, 0],
            )
        elif transform_id == "TranslateX":
            return F.affine(
                image,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill_,
            )
        elif transform_id == "TranslateY":
            return F.affine(
                image,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill_,
            )
        elif transform_id == "Rotate":
            return F.rotate(image, angle=magnitude, interpolation=interpolation, fill=fill_)
        elif transform_id == "Brightness":
            return F.adjust_brightness(image, brightness_factor=1.0 + magnitude)
        elif transform_id == "Color":
            return F.adjust_saturation(image, saturation_factor=1.0 + magnitude)
        elif transform_id == "Contrast":
            return F.adjust_contrast(image, contrast_factor=1.0 + magnitude)
        elif transform_id == "Sharpness":
            return F.adjust_sharpness(image, sharpness_factor=1.0 + magnitude)
        elif transform_id == "Posterize":
            return F.posterize(image, bits=int(magnitude))
        elif transform_id == "Solarize":
            return F.solarize(image, threshold=magnitude)
        elif transform_id == "AutoContrast":
            return F.autocontrast(image)
        elif transform_id == "Equalize":
            return F.equalize(image)
        elif transform_id == "Invert":
            return F.invert(image)
        else:
            raise ValueError(f"No transform available for {transform_id}")


class AutoAugment(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins),
            True,
        ),
        "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (
            lambda num_bins, height, width: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, height, width: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
        "Invert": (lambda num_bins, height, width: None, False),
    }

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
    ) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.policy = policy
        self._policies = self._get_policies(policy)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        id, image = self._extract_image(sample)
        num_channels, height, width = get_chw(image)

        policy = self._policies[int(torch.randint(len(self._policies), ()))]

        for transform_id, probability, magnitude_idx in policy:
            if not torch.rand(()) <= probability:
                continue

            magnitudes_fn, signed = self._AUGMENTATION_SPACE[transform_id]

            magnitudes = magnitudes_fn(10, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[magnitude_idx])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0

            image = self._apply_image_transform(
                image, transform_id, magnitude, interpolation=self.interpolation, fill=self.fill
            )

        return self._put_into_sample(sample, id, image)


class RandAugment(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins),
            True,
        ),
        "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (
            lambda num_bins, height, width: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, height, width: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
    ) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        id, image = self._extract_image(sample)
        _, height, width = get_chw(image)

        for _ in range(self.num_ops):
            transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)
            magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[self.magnitude])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0
            image = self._apply_image_transform(
                image, transform_id, magnitude, interpolation=self.interpolation, fill=self.fill
            )

        return self._put_into_sample(sample, id, image)


class TrivialAugmentWide(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
        "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
        "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Posterize": (
            lambda num_bins, height, width: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, height, width: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
    ):
        super().__init__(interpolation=interpolation, fill=fill)
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        id, image = self._extract_image(sample)
        _, height, width = get_chw(image)

        transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)

        magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0

        image = self._apply_image_transform(
            image, transform_id, magnitude, interpolation=self.interpolation, fill=self.fill
        )
        return self._put_into_sample(sample, id, image)


class AugMix(_AutoAugmentBase):
    _PARTIAL_AUGMENTATION_SPACE = {
        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, width / 3.0, num_bins), True),
        "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, height / 3.0, num_bins), True),
        "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
        "Posterize": (
            lambda num_bins, height, width: cast(torch.Tensor, 4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, height, width: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }
    _AUGMENTATION_SPACE: Dict[str, Tuple[Callable[[int, int, int], Optional[torch.Tensor]], bool]] = {
        **_PARTIAL_AUGMENTATION_SPACE,
        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
    }

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
    ) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self._PARAMETER_MAX = 10
        if not (1 <= severity <= self._PARAMETER_MAX):
            raise ValueError(f"The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead.")
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops

    def _sample_dirichlet(self, params: torch.Tensor) -> torch.Tensor:
        # Must be on a separate method so that we can overwrite it in tests.
        return torch._sample_dirichlet(params)

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        id, orig_image = self._extract_image(sample)
        _, height, width = get_chw(orig_image)

        if isinstance(orig_image, torch.Tensor):
            image = orig_image
        else:  # isinstance(inpt, PIL.Image.Image):
            image = F.to_image_tensor(orig_image)

        augmentation_space = self._AUGMENTATION_SPACE if self.all_ops else self._PARTIAL_AUGMENTATION_SPACE

        orig_dims = list(image.shape)
        batch = image.view([1] * max(4 - image.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

        # Sample the beta weights for combining the original and augmented image. To get Beta, we use a Dirichlet
        # with 2 parameters. The 1st column stores the weights of the original and the 2nd the ones of augmented image.
        m = self._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1)
        )

        # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented images.
        combined_weights = self._sample_dirichlet(
            torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)
        ) * m[:, 1].view([batch_dims[0], -1])

        mix = m[:, 0].view(batch_dims) * batch
        for i in range(self.mixture_width):
            aug = batch
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            for _ in range(depth):
                transform_id, (magnitudes_fn, signed) = self._get_random_item(augmentation_space)

                magnitudes = magnitudes_fn(self._PARAMETER_MAX, height, width)
                if magnitudes is not None:
                    magnitude = float(magnitudes[int(torch.randint(self.severity, ()))])
                    if signed and torch.rand(()) <= 0.5:
                        magnitude *= -1
                else:
                    magnitude = 0.0

                aug = self._apply_image_transform(
                    aug, transform_id, magnitude, interpolation=self.interpolation, fill=self.fill
                )
            mix.add_(combined_weights[:, i].view(batch_dims) * aug)
        mix = mix.view(orig_dims).to(dtype=image.dtype)

        if isinstance(orig_image, features.Image):
            mix = features.Image.new_like(orig_image, mix)
        elif isinstance(orig_image, PIL.Image.Image):
            mix = F.to_image_pil(mix)

        return self._put_into_sample(sample, id, mix)
