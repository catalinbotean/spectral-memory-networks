from torch import Tensor
import torch.nn.functional as F


def multiclass_dice_score(
    logits: Tensor,
    ground_truth: Tensor,
    epsilon: float = 1e-7,
) -> float:
    """
    Computes the multi-class Dice score for semantic segmentation.

    The Dice coefficient for class k is defined as:
        Dice_k = 2 * |P_k ∩ T_k| / (|P_k| + |T_k|)

    The returned metric is the average Dice score across all classes and
    samples in the batch.
    """
    _validate_multiclass_dice_input_shapes(logits, ground_truth)

    probability_distributions = _convert_logits_to_probability_distributions(logits)
    ground_truth_one_hot = _convert_ground_truth_to_one_hot_encoding(
        ground_truth,
        number_of_classes=probability_distributions.size(1),
    )

    per_class_intersection = _compute_per_class_intersection(
        probability_distributions,
        ground_truth_one_hot,
    )
    per_class_union = _compute_per_class_union(
        probability_distributions,
        ground_truth_one_hot,
    )

    dice_scores_per_class = _compute_per_class_dice_scores(
        per_class_intersection,
        per_class_union,
        epsilon,
    )

    return _compute_mean_dice_score_over_batch_and_classes(dice_scores_per_class)


# -------------------------------------------------------------------------
# Helper functions (academic naming)
# -------------------------------------------------------------------------

def _validate_multiclass_dice_input_shapes(logits: Tensor, ground_truth: Tensor) -> None:
    if logits.ndim != 4:
        raise ValueError(
            f"'logits' must have shape [batch_size, num_classes, height, width], "
            f"but got {tuple(logits.shape)}."
        )
    if ground_truth.ndim != 3:
        raise ValueError(
            f"'ground_truth' must have shape [batch_size, height, width], "
            f"but got {tuple(ground_truth.shape)}."
        )
    if logits.size(0) != ground_truth.size(0):
        raise ValueError(
            f"Batch size mismatch: logits batch={logits.size(0)}, "
            f"ground_truth batch={ground_truth.size(0)}."
        )


def _convert_logits_to_probability_distributions(logits: Tensor) -> Tensor:
    """Apply a softmax transformation to produce per-class probability distributions."""
    return F.softmax(logits, dim=1)


def _convert_ground_truth_to_one_hot_encoding(
    ground_truth: Tensor,
    number_of_classes: int,
) -> Tensor:
    """Convert class index labels [B,H,W] into one-hot encoded form [B,C,H,W]."""
    return (
        F.one_hot(ground_truth, number_of_classes)
        .permute(0, 3, 1, 2)
        .float()
    )


def _compute_per_class_intersection(
    probability_distributions: Tensor,
    ground_truth_one_hot: Tensor,
) -> Tensor:
    """Compute the intersection between prediction and ground truth for each class."""
    return (probability_distributions * ground_truth_one_hot).sum(dim=(2, 3))


def _compute_per_class_union(
    probability_distributions: Tensor,
    ground_truth_one_hot: Tensor,
) -> Tensor:
    """Compute the union term for each class."""
    return (
        probability_distributions.sum(dim=(2, 3))
        + ground_truth_one_hot.sum(dim=(2, 3))
    )


def _compute_per_class_dice_scores(
    per_class_intersection: Tensor,
    per_class_union: Tensor,
    epsilon: float,
) -> Tensor:
    """Compute Dice score for each class and each sample."""
    return (2.0 * per_class_intersection) / (per_class_union + epsilon)


def _compute_mean_dice_score_over_batch_and_classes(
    dice_scores: Tensor,
) -> float:
    """Compute the final mean Dice score."""
    return dice_scores.mean().item()
