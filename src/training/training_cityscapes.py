from pathlib import Path

from torch.optim import AdamW

from src.data.cityscapes import create_cityscapes_dataloaders
from src.models.spectral_memory_unet import SpectralMemoryUNet
from src.training.trainer_cityscapes import CityscapesTrainer, TrainerConfig


def main() -> None:
    root = Path("datasets/cityscapes")  # adapt to your path

    train_loader, val_loader = create_cityscapes_dataloaders(
        root_directory=root,
        batch_size=2,
        num_workers=4,
        resize_to=(512, 1024),
    )

    model = SpectralMemoryUNet(
        input_channels=3,
        output_channels=19,   # Cityscapes has 19 classes
        number_of_encoder_stages=4,
        base_number_of_channels=32,
        memory_length=2,
    )

    config = TrainerConfig(
        learning_rate=1e-4,
        weight_decay=1e-5,
        max_epochs=150,
        num_classes=19,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    trainer = CityscapesTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
