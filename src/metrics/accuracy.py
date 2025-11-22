import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Top-1 accuracy for classification.

    Args:
        logits: Tensor of shape [B, C].
        targets: Tensor of shape [B] with class indices.

    Returns:
        Accuracy in [0, 1] as a Python float.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)
