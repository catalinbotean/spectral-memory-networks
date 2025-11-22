import torch
from torch import Tensor


def depth_absolute_relative_error(
    predicted_depth: Tensor,
    ground_truth_depth: Tensor,
    epsilon: float = 1e-6,
) -> float:
    """
    Computes the Absolute Relative Error (AbsRel) metric for depth estimation.

    Definition:
        AbsRel = mean( |predicted_depth - ground_truth_depth| /
                       max(ground_truth_depth, epsilon) )

    Args:
        predicted_depth:
            Tensor of shape [batch_size, 1, height, width] or [batch_size, height, width].
        ground_truth_depth:
            Tensor with identical shape.
        epsilon:
            Minimum value used to avoid division by zero.

    Returns:
        A Python float representing the absolute relative error.
    """
    _validate_depth_absolute_relative_error_input_shapes(
        predicted_depth,
        ground_truth_depth,
    )

    predicted_depth = _convert_depth_tensor_to_float(predicted_depth)
    ground_truth_depth = _convert_depth_tensor_to_float(ground_truth_depth)

    stabilized_ground_truth = _apply_minimum_value_epsilon(
        ground_truth_depth,
        epsilon,
    )

    absolute_difference = _compute_absolute_depth_difference(
        predicted_depth,
        stabilized_ground_truth,
    )

    relative_error = _compute_relative_depth_error(
        absolute_difference,
        stabilized_ground_truth,
    )

    return _compute_mean_absolute_relative_error(relative_error)


# -------------------------------------------------------------------------
# Helper functions — academic naming
# -------------------------------------------------------------------------

def _validate_depth_absolute_relative_error_input_shapes(
    predicted_depth: Tensor,
    ground_truth_depth: Tensor,
) -> None:
    """
    Validates that both predicted and ground-truth depth tensors have the same shape
    and belong to one of the accepted dimensional formats.
    """
    if predicted_depth.shape != ground_truth_depth.shape:
        raise ValueError(
            "Shape mismatch: 'predicted_depth' and 'ground_truth_depth' must match. "
            f"Received predicted={tuple(predicted_depth.shape)}, "
            f"ground_truth={tuple(ground_truth_depth.shape)}."
        )

    if predicted_depth.ndim not in (3, 4):
        raise ValueError(
            "'predicted_depth' must have shape [batch_size, height, width] "
            "or [batch_size, 1, height, width]. "
            f"Received shape: {tuple(predicted_depth.shape)}."
        )


def _convert_depth_tensor_to_float(depth_tensor: Tensor) -> Tensor:
    """
    Converts a depth tensor to floating point precision for numerical stability.
    """
    return depth_tensor.float()


def _apply_minimum_value_epsilon(
    ground_truth_depth: Tensor,
    epsilon: float,
) -> Tensor:
    """
    Ensures that ground-truth depth values are not below `epsilon`,
    preventing division-by-zero issues.
    """
    return ground_truth_depth.clamp(min=epsilon)


def _compute_absolute_depth_difference(
    predicted_depth: Tensor,
    ground_truth_depth: Tensor,
) -> Tensor:
    """
    Computes the absolute difference |predicted_depth - ground_truth_depth|.
    """
    return torch.abs(predicted_depth - ground_truth_depth)


def _compute_relative_depth_error(
    absolute_difference: Tensor,
    ground_truth_depth: Tensor,
) -> Tensor:
    """
    Computes the relative depth error: |pred - gt| / gt.
    """
    return absolute_difference / ground_truth_depth


def _compute_mean_absolute_relative_error(relative_error: Tensor) -> float:
    """
    Computes the mean absolute relative error over all pixels and returns
    it as a Python float.
    """
    return relative_error.mean().item()
