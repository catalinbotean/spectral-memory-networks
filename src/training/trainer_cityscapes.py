from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.metrics.mean_intersection_over_union import mean_intersection_over_union
from src.metrics.multiclass_dice_score import multiclass_dice_score
from src.utils.logging import SimpleLogger


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes: int = 19
    log_every: int = 10
    log_dir: str = "./logs"


class CityscapesTrainer:
    """
    Trainer for semantic segmentation on the Cityscapes dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        config: TrainerConfig,
    ) -> None:
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.logger = SimpleLogger(log_dir=config.log_dir, name="cityscapes_training")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _train_step(self, batch: Dict[str, Tensor]) -> float:
        self.model.train()

        images = batch["image"].to(self.config.device)
        masks = batch["mask"].to(self.config.device)

        logits = self.model(images)
        loss = self.criterion(logits, masks)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def _val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()

        with torch.no_grad():
            images = batch["image"].to(self.config.device)
            masks = batch["mask"].to(self.config.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            miou = mean_intersection_over_union(
                logits=logits,
                ground_truth=masks,
                num_classes=self.config.num_classes,
            )

            dice = multiclass_dice_score(
                logits=logits,
                ground_truth=masks,
            )

        return {
            "loss": float(loss.item()),
            "miou": miou,
            "dice": dice,
        }

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        self.logger.log(f"Starting training for {self.config.max_epochs} epochs on {self.config.device}")
        self.logger.log(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(1, self.config.max_epochs + 1):
            train_losses = []

            for step_index, batch in enumerate(self.train_loader):
                loss = self._train_step(batch)
                train_losses.append(loss)

                if step_index % self.config.log_every == 0:
                    self.logger.log(
                        f"[Epoch {epoch:03d}] Step {step_index:04d} "
                        f"Train Loss: {loss:.4f}"
                    )

            avg_train_loss = sum(train_losses) / len(train_losses)

            val_metrics = self._run_validation_epoch()

            self.logger.log(
                f"==== Epoch {epoch}/{self.config.max_epochs} ====\n"
                f"Train Loss: {avg_train_loss:.4f}\n"
                f"Val Loss:   {val_metrics['loss']:.4f}\n"
                f"mIoU:       {val_metrics['miou']:.4f}\n"
                f"Dice:       {val_metrics['dice']:.4f}"
            )

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------
    def _run_validation_epoch(self) -> Dict[str, float]:
        val_losses = []
        miou_values = []
        dice_values = []

        for batch in self.val_loader:
            metrics = self._val_step(batch)
            val_losses.append(metrics["loss"])
            miou_values.append(metrics["miou"])
            dice_values.append(metrics["dice"])

        return {
            "loss": sum(val_losses) / len(val_losses),
            "miou": sum(miou_values) / len(miou_values),
            "dice": sum(dice_values) / len(dice_values),
        }
