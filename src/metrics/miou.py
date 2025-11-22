import torch


def miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Mean Intersection-over-Union (mIoU) for semantic segmentation.

    Args:
        preds: Logits of shape [B, C, H, W].
        targets: Ground truth tensor of shape [B, H, W] with values in [0, C-1].
        num_classes: Number of classes C.

    Returns:
        mIoU over present classes as a Python float.
    """
    preds_labels = preds.argmax(dim=1)
    ious = []

    for cls in range(num_classes):
        pred_mask = (preds_labels == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)
