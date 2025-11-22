import torch
import torch.nn.functional as F


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """
    Multi-class Dice score for semantic segmentation.

    Args:
        preds: Logits of shape [B, C, H, W].
        targets: Ground truth of shape [B, H, W] with values in [0, C-1].
        eps: Small constant for numerical stability.

    Returns:
        Mean Dice score over classes and batch as a Python float.
    """
    probs = F.softmax(preds, dim=1)
    B, C, H, W = probs.shape

    targets_1h = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

    intersect = (probs * targets_1h).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets_1h.sum(dim=(2, 3))

    dice = 2 * intersect / (union + eps)
    return dice.mean().item()
