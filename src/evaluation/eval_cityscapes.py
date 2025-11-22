from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.cityscapes import CityscapesSegmentationDataset, CityscapesNormalizationConfig
from src.models.spectral_memory_unet import SpectralMemoryUNet
from src.evaluation.cityscapes_evaluation import CityscapesEvaluator, CityscapesEvalConfig
from src.utils.logging import SimpleLogger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SpectralMemoryUNet on Cityscapes.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory where Cityscapes is stored.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=512,
        help="Resize height for images and masks.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=1024,
        help="Resize width for images and masks.",
    )
    parser.add_argument(
        "--save-predictions-dir",
        type=str,
        default=None,
        help="Optional directory to save predicted segmentation maps.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on number of batches for quick evaluation.",
    )

    return parser.parse_args()


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """
    Load a model checkpoint.

    Supports both:
      - state_dict saved directly (torch.save(model.state_dict())),
      - wrapped dict with key 'model_state'.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)


def build_dataloader(
    root_directory: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    resize_height: int,
    resize_width: int,
) -> DataLoader:
    dataset = CityscapesSegmentationDataset(
        root_directory=root_directory,
        split=split,
        resize_to=(resize_height, resize_width),
        use_random_horizontal_flip=False,
        normalization_config=CityscapesNormalizationConfig(),
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def main() -> None:
    arguments = parse_arguments()

    # Initialize logger for evaluation results
    logger = SimpleLogger(log_dir="./evaluation_logs", name="cityscapes_evaluation")

    data_root = Path(arguments.data_root)
    checkpoint_path = Path(arguments.checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loader = build_dataloader(
        root_directory=data_root,
        split=arguments.split,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        resize_height=arguments.resize_height,
        resize_width=arguments.resize_width,
    )

    model = SpectralMemoryUNet(
        input_channels=3,
        output_channels=19,  # Cityscapes: 19 semantic classes
        number_of_encoder_stages=4,
        base_number_of_channels=32,
        memory_length=2,
    )

    _load_checkpoint(model, checkpoint_path)

    eval_config = CityscapesEvalConfig(
        device=device,
        num_classes=19,
        ignore_index=255,
        save_predictions_directory=(
            Path(arguments.save_predictions_dir)
            if arguments.save_predictions_dir is not None
            else None
        ),
        max_batches=arguments.max_batches,
    )

    evaluator = CityscapesEvaluator(
        model=model,
        data_loader=data_loader,
        config=eval_config,
    )

    metrics = evaluator.evaluate()

    logger.log("==== Cityscapes Evaluation ====")
    logger.log(f"Split: {arguments.split}")
    logger.log(f"Checkpoint: {checkpoint_path}")
    logger.log(f"Loss: {metrics['loss']:.4f}")
    logger.log(f"mIoU: {metrics['miou']:.4f}")
    logger.log(f"Dice: {metrics['dice']:.4f}")


if __name__ == "__main__":
    main()
