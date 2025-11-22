import torch


def depth_rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Root Mean Squared Error for depth estimation.

    Args:
        pred: Predicted depth, shape [B, 1, H, W] (or [B, H, W]).
        gt: Ground truth depth, same shape as pred.

    Returns:
        RMSE as a Python float.
    """
    pred = pred.float()
    gt = gt.float()

    return torch.sqrt(((pred - gt) ** 2).mean()).item()
