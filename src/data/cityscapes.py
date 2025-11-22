from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes as TorchvisionCityscapes
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


CityscapesTransform = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


@dataclass
class CityscapesNormalizationConfig:
    """
    Configuration for image normalization.

    Default values correspond to ImageNet statistics, which are commonly
    used for natural image datasets.
    """
    mean_red: float = 0.485
    mean_green: float = 0.456
    mean_blue: float = 0.406
    std_red: float = 0.229
    std_green: float = 0.224
    std_blue: float = 0.225

    def as_tensors(self) -> Tuple[Tensor, Tensor]:
        mean = torch.tensor(
            [self.mean_red, self.mean_green, self.mean_blue],
            dtype=torch.float32,
        )
        std = torch.tensor(
            [self.std_red, self.std_green, self.std_blue],
            dtype=torch.float32,
        )
        return mean, std


class CityscapesSegmentationDataset(Dataset):
    """
    Wrapper around torchvision's Cityscapes dataset for semantic segmentation.

    This class:
      - loads RGB images and semantic segmentation masks,
      - optionally resizes them to a fixed spatial resolution,
      - applies joint augmentations on image and mask,
      - normalizes images according to a configurable scheme.

    Expected directory layout (as used by torchvision.datasets.Cityscapes):

        root/
          └── cityscapes/
                ├── leftImg8bit/
                │     ├── train/
                │     ├── val/
                │     └── test/
                └── gtFine/
                      ├── train/
                      ├── val/
                      └── test/
    """

    def __init__(
        self,
        root_directory: Path | str,
        split: str = "train",
        resize_to: Optional[Tuple[int, int]] = None,
        use_random_horizontal_flip: bool = False,
        normalization_config: Optional[CityscapesNormalizationConfig] = None,
        ignore_label: int = 255,
    ) -> None:
        super().__init__()

        self.root_directory = Path(root_directory)
        self.split = self._validate_split(split)
        self.resize_to = resize_to
        self.use_random_horizontal_flip = use_random_horizontal_flip
        self.normalization_config = (
            normalization_config
            if normalization_config is not None
            else CityscapesNormalizationConfig()
        )
        self.ignore_label = ignore_label

        self._cityscapes_dataset = self._build_torchvision_cityscapes_dataset()

    # -------------------------------------------------------------------------
    # Dataset construction
    # -------------------------------------------------------------------------

    def _build_torchvision_cityscapes_dataset(self) -> TorchvisionCityscapes:
        """
        Construct the underlying torchvision Cityscapes dataset instance.
        """
        return TorchvisionCityscapes(
            root=str(self.root_directory),
            split=self.split,
            mode="fine",
            target_type="semantic",
        )

    @staticmethod
    def _validate_split(split: str) -> str:
        """
        Ensure that the requested split is one of the supported values.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split '{split}'. Supported splits are: {valid_splits}."
            )
        return split

    # -------------------------------------------------------------------------
    # Length and indexing
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cityscapes_dataset)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """
        Returns:
            A dictionary with keys:
              - "image": Tensor [3, H, W], normalized to [0, 1] then standardized.
              - "mask": Tensor [H, W] with integer labels.
        """
        image, mask = self._cityscapes_dataset[index]

        image_tensor = self._convert_image_to_tensor(image)
        mask_tensor = self._convert_mask_to_tensor(mask)

        image_tensor, mask_tensor = self._apply_joint_spatial_transforms(
            image_tensor,
            mask_tensor,
        )

        image_tensor = self._normalize_image(image_tensor)

        sample: Dict[str, Tensor] = {
            "image": image_tensor,
            "mask": mask_tensor,
        }
        return sample

    # -------------------------------------------------------------------------
    # Conversion helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _convert_image_to_tensor(image) -> Tensor:
        """
        Convert a PIL image to a float32 tensor in [0, 1].
        """
        return F.to_tensor(image)

    @staticmethod
    def _convert_mask_to_tensor(mask) -> Tensor:
        """
        Convert a PIL mask to a long tensor of shape [H, W].
        """
        mask_tensor = torch.as_tensor(mask, dtype=torch.int64)
        return mask_tensor

    # -------------------------------------------------------------------------
    # Spatial transforms (joint on image and mask)
    # -------------------------------------------------------------------------

    def _apply_joint_spatial_transforms(
        self,
        image: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Applies spatial transforms that must be consistent between image and mask,
        such as resizing and horizontal flipping.
        """
        if self.resize_to is not None:
            image, mask = self._resize_image_and_mask(
                image=image,
                mask=mask,
                size=self.resize_to,
            )

        if self.use_random_horizontal_flip:
            image, mask = self._random_horizontal_flip(
                image=image,
                mask=mask,
            )

        return image, mask

    @staticmethod
    def _resize_image_and_mask(
        image: Tensor,
        mask: Tensor,
        size: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Resize image and mask to the same spatial resolution.

        Args:
            image: Tensor [3, H, W].
            mask: Tensor [H, W].
            size: (height, width).
        """
        resized_image = F.resize(
            image,
            size=size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        resized_mask = F.resize(
            mask.unsqueeze(0).float(),
            size=size,
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0).long()

        return resized_image, resized_mask

    @staticmethod
    def _random_horizontal_flip(
        image: Tensor,
        mask: Tensor,
        probability: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Random horizontal flip applied consistently to image and mask.
        """
        if torch.rand(1).item() < probability:
            image = torch.flip(image, dims=[2])  # width dimension
            mask = torch.flip(mask, dims=[1])
        return image, mask

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def _normalize_image(self, image: Tensor) -> Tensor:
        """
        Normalize image using the configured mean and standard deviation.
        """
        mean, std = self.normalization_config.as_tensors()
        return F.normalize(image, mean=mean, std=std)


# -------------------------------------------------------------------------
# Dataloader utilities
# -------------------------------------------------------------------------


def create_cityscapes_dataloaders(
    root_directory: Path | str,
    batch_size: int = 4,
    num_workers: int = 4,
    resize_to: Optional[Tuple[int, int]] = (512, 1024),
    use_random_horizontal_flip_for_training: bool = True,
    normalization_config: Optional[CityscapesNormalizationConfig] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function for building training and validation dataloaders
    for Cityscapes semantic segmentation.

    Args:
        root_directory:
            Root directory where the Cityscapes dataset is stored.
        batch_size:
            Batch size used for both training and validation loaders.
        num_workers:
            Number of worker processes for data loading.
        resize_to:
            Optional spatial resolution (height, width) to which all images
            and masks are resized.
        use_random_horizontal_flip_for_training:
            Whether to apply random horizontal flipping on the training split.
        normalization_config:
            Configuration object defining the mean and standard deviation used
            for image normalization.

    Returns:
        (train_loader, validation_loader)
    """
    training_dataset = CityscapesSegmentationDataset(
        root_directory=root_directory,
        split="train",
        resize_to=resize_to,
        use_random_horizontal_flip=use_random_horizontal_flip_for_training,
        normalization_config=normalization_config,
    )

    validation_dataset = CityscapesSegmentationDataset(
        root_directory=root_directory,
        split="val",
        resize_to=resize_to,
        use_random_horizontal_flip=False,
        normalization_config=normalization_config,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return training_loader, validation_loader
