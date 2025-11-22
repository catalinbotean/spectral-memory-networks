import torch
from torch import Tensor


def depth_root_mean_squared_error(
    predicted_depth: Tensor,
    ground_truth_depth: Tensor,
) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) for depth estimation.

    Definition:
        RMSE = sqrt( mean( (predicted_depth - ground_truth_depth)^2 ) )

    Args:
        predicted_depth:
            Tensor of shape [batch_size, 1, height, width] or [batch_size, height, width].
        ground_truth_depth:
            Tensor with identical shape to `predicted_depth`.

    Returns:
        A Python float representing the RMSE value.
    """
    _validate_depth_root_mean_squared_error_input_shapes(
        predicted_depth,
        ground_truth_depth,
    )

    predicted_depth = _convert_depth_tensor_to_float(predicted_depth)
    ground_truth_depth = _convert_depth_tensor_to_float(ground_truth_depth)

    squared_difference = _compute_squared_depth_difference(
        predicted_depth,
        ground_truth_depth,
    )
    rmse_value = _compute_root_mean_squared_error_from_squared_difference(
        squared_difference
    )

    return rmse_value


# -------------------------------------------------------------------------
# Helper functions — academic naming
# -------------------------------------------------------------------------

def _validate_depth_root_mean_squared_error_input_shapes(
    predicted_depth: Tensor,
    ground_truth_depth: Tensor,
) -> None:
    """
    Ensures that the predicted and ground-truth depth tensors have identical shapes.
    """
    if predicted_depth.shape != ground_truth_depth.shape:
        raise ValueError(
            "Input shape mismatch: 'predicted_depth' and 'ground_truth_depth' "
            f"must have identical shapes, but received "
            f"{tuple(predicted_depth.shape)} and {tuple(ground_truth_depth.shape)}."
        )

    if predicted_depth.ndim not in (3, 4):
        raise ValueError(
            "'predicted_depth' must have either 3 dimensions [B, H, W] "
            "or 4 dimensions [B, 1, H, W]. "
            f"Received tensor with shape: {tuple(predicted_depth.shape)}."
        )


def _convert_depth_tensor_to_float(depth_tensor: Tensor) -> Tensor:
    """
    Converts a depth tensor to floating point precision (32-bit).
    """
    return depth_tensor.float()


def _compute_squared_depth_difference(
    predicted_depth: Tensor,
    ground_truth_depth: Tensor,
) -> Tensor:
    """
    Computes the element-wise squared difference between predicted and true depth maps.
    """
    return (predicted_depth - ground_truth_depth) ** 2


def _compute_root_mean_squared_error_from_squared_difference(
    squared_difference: Tensor,
) -> float:
    """
    Computes the root mean squared error (RMSE) from a squared difference tensor.
    """
    mean_squared_difference = squared_difference.mean()
    return torch.sqrt(mean_squared_difference).item()
