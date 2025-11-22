from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.metrics.mean_intersection_over_union import mean_intersection_over_union
from src.metrics.multiclass_dice_score import multiclass_dice_score 


@dataclass
class CityscapesEvalConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes: int = 19
    ignore_index: int = 255
    save_predictions_directory: Optional[Path] = None
    max_batches: Optional[int] = None  # for quick sanity eval, None = eval all


class CityscapesEvaluator:
    """
    Evaluator for semantic segmentation models on the Cityscapes dataset.

    Computes:
      - Cross-entropy loss (with ignore_index),
      - mean IoU,
      - multi-class Dice score.

    Optionally saves predicted segmentation masks to disk.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        config: CityscapesEvalConfig,
    ) -> None:
        self.model = model.to(config.device)
        self.data_loader = data_loader
        self.config = config

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)

        if (
            self.config.save_predictions_directory is not None
        ):
            self.config.save_predictions_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation over the entire dataloader (or a limited number of batches).

        Returns:
            Dictionary with keys: "loss", "miou", "dice".
        """
        self.model.eval()

        total_loss = 0.0
        total_miou = 0.0
        total_dice = 0.0
        number_of_batches = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(self.data_loader):
                if (
                    self.config.max_batches is not None
                    and batch_index >= self.config.max_batches
                ):
                    break

                loss_value, miou_value, dice_value = self._evaluate_single_batch(
                    batch_index,
                    batch,
                )

                total_loss += loss_value
                total_miou += miou_value
                total_dice += dice_value
                number_of_batches += 1

        if number_of_batches == 0:
            raise RuntimeError("No batches were processed during evaluation.")

        return {
            "loss": total_loss / number_of_batches,
            "miou": total_miou / number_of_batches,
            "dice": total_dice / number_of_batches,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_single_batch(
        self,
        batch_index: int,
        batch: Dict[str, Tensor],
    ) -> tuple[float, float, float]:
        images = batch["image"].to(self.config.device)
        masks = batch["mask"].to(self.config.device)

        logits = self.model(images)
        loss = self.criterion(logits, masks)

        miou_value = mean_intersection_over_union(
            logits=logits,
            ground_truth=masks,
            num_classes=self.config.num_classes,
        )

        dice_value = multiclass_dice_score(
            logits=logits,
            ground_truth=masks,
        )

        if self.config.save_predictions_directory is not None:
            self._save_predictions_for_batch(
                batch_index=batch_index,
                logits=logits,
            )

        return float(loss.item()), miou_value, dice_value

    def _save_predictions_for_batch(
        self,
        batch_index: int,
        logits: Tensor,
    ) -> None:
        """
        Optionally save predicted segmentation maps as torch tensors on disk.

        You can later convert them to color PNGs in a separate script if desired.
        """
        predictions = torch.argmax(logits, dim=1).cpu()  # [B, H, W]

        for sample_index in range(predictions.size(0)):
            prediction = predictions[sample_index]

            file_name = f"batch{batch_index:04d}_sample{sample_index:02d}.pt"
            file_path = self.config.save_predictions_directory / file_name

            torch.save(prediction, file_path)
