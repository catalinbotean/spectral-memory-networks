import torch
from torch import Tensor
from typing import List


def mean_intersection_over_union(
    logits: Tensor,
    ground_truth: Tensor,
    number_of_classes: int,
) -> float:
    """
    Computes the mean Intersection-over-Union (mIoU) for semantic segmentation.

    For each semantic class k in {0, ..., C-1}:
        IoU_k = |P_k ∩ T_k| / |P_k ∪ T_k|

    The final metric is the arithmetic mean of IoU_k over all classes for which
    the union |P_k ∪ T_k| is non-zero (i.e., at least one pixel is present in
    either prediction or ground truth for that class).

    Args:
        logits:
            Tensor of shape [batch_size, number_of_classes, height, width]
            containing raw class scores (logits) for each pixel.
        ground_truth:
            Tensor of shape [batch_size, height, width] containing integer
            class labels in [0, number_of_classes - 1].
        number_of_classes:
            The total number of semantic classes.

    Returns:
        A Python float representing the mean Intersection-over-Union over all
        valid classes.
    """
    _validate_mean_intersection_over_union_inputs(
        logits=logits,
        ground_truth=ground_truth,
        number_of_classes=number_of_classes,
    )

    predicted_class_labels = _convert_logits_to_predicted_class_labels(logits)
    per_class_intersection_over_union = _compute_per_class_intersection_over_union(
        predicted_class_labels=predicted_class_labels,
        ground_truth=ground_truth,
        number_of_classes=number_of_classes,
    )

    return _compute_mean_intersection_over_union_over_valid_classes(
        per_class_intersection_over_union
    )


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


def _validate_mean_intersection_over_union_inputs(
    logits: Tensor,
    ground_truth: Tensor,
    number_of_classes: int,
) -> None:
    if logits.ndim != 4:
        raise ValueError(
            "'logits' must have shape [batch_size, number_of_classes, height, width], "
            f"but got {tuple(logits.shape)}."
        )

    if ground_truth.ndim != 3:
        raise ValueError(
            "'ground_truth' must have shape [batch_size, height, width], "
            f"but got {tuple(ground_truth.shape)}."
        )

    if logits.size(0) != ground_truth.size(0):
        raise ValueError(
            "The batch dimension of 'logits' and 'ground_truth' must match. "
            f"Got logits batch={logits.size(0)}, ground_truth batch={ground_truth.size(0)}."
        )

    if logits.size(1) != number_of_classes:
        raise ValueError(
            "The channel dimension of 'logits' must be equal to 'number_of_classes'. "
            f"Got logits channels={logits.size(1)}, number_of_classes={number_of_classes}."
        )


def _convert_logits_to_predicted_class_labels(logits: Tensor) -> Tensor:
    """
    Converts per-class logits [B, C, H, W] into discrete class predictions [B, H, W]
    by taking the argmax over the class dimension.
    """
    return logits.argmax(dim=1)


def _compute_per_class_intersection_over_union(
    predicted_class_labels: Tensor,
    ground_truth: Tensor,
    number_of_classes: int,
) -> List[float]:
    """
    Computes IoU for each class independently and returns a list containing
    IoU_k values for all classes with non-zero union.
    """
    per_class_ious: List[float] = []

    for class_index in range(number_of_classes):
        predicted_mask_for_class = predicted_class_labels == class_index
        ground_truth_mask_for_class = ground_truth == class_index

        intersection = _compute_binary_mask_intersection(
            predicted_mask_for_class,
            ground_truth_mask_for_class,
        )
        union = _compute_binary_mask_union(
            predicted_mask_for_class,
            ground_truth_mask_for_class,
        )

        if union == 0:
            # Class is absent both in prediction and ground truth.
            continue

        per_class_ious.append(intersection / union)

    return per_class_ious


def _compute_binary_mask_intersection(
    predicted_mask: Tensor,
    ground_truth_mask: Tensor,
) -> int:
    """
    Computes the size of the intersection of two boolean masks as an integer.
    """
    return (predicted_mask & ground_truth_mask).sum().item()


def _compute_binary_mask_union(
    predicted_mask: Tensor,
    ground_truth_mask: Tensor,
) -> int:
    """
    Computes the size of the union of two boolean masks as an integer.
    """
    return (predicted_mask | ground_truth_mask).sum().item()


def _compute_mean_intersection_over_union_over_valid_classes(
    per_class_intersection_over_union: list[float],
) -> float:
    """
    Computes the arithmetic mean of IoU values over all valid classes.
    If no class has a non-zero union, returns 0.0.
    """
    if not per_class_intersection_over_union:
        return 0.0

    return sum(per_class_intersection_over_union) / len(per_class_intersection_over_union)
