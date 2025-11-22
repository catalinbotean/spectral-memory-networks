import torch


def depth_abs_rel(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Absolute Relative Error for depth estimation.

    Args:
        pred: Predicted depth, shape [B, 1, H, W] (or [B, H, W]).
        gt: Ground truth depth, same shape as pred.

    Returns:
        Absolute relative error as a Python float.
    """
    pred = pred.float()
    gt = gt.float().clamp(min=1e-6)

    return torch.mean(torch.abs(pred - gt) / gt).item()
