from torch import Tensor


def classification_accuracy(
    logits: Tensor,
    ground_truth_class_indices: Tensor,
) -> float:
    """
    Computes the top-1 classification accuracy.

    Definition:
        accuracy = mean( predicted_class == ground_truth_class )

    Args:
        logits:
            Tensor of shape [batch_size, number_of_classes] containing raw logits.
        ground_truth_class_indices:
            Tensor of shape [batch_size] with integer class labels.

    Returns:
        A Python float in the range [0.0, 1.0] representing the classification accuracy.
    """
    _validate_classification_accuracy_input_shapes(
        logits,
        ground_truth_class_indices,
    )

    predicted_class_indices = _convert_logits_to_predicted_class_indices(logits)
    correct_prediction_mask = _compute_correct_prediction_mask(
        predicted_class_indices,
        ground_truth_class_indices,
    )

    return _compute_mean_classification_accuracy(correct_prediction_mask)


# -------------------------------------------------------------------------
# Helper functions — academic naming
# -------------------------------------------------------------------------

def _validate_classification_accuracy_input_shapes(
    logits: Tensor,
    ground_truth_class_indices: Tensor,
) -> None:
    """
    Ensures that the logits and ground-truth class index tensors have valid shapes.
    """
    if logits.ndim != 2:
        raise ValueError(
            "'logits' must have shape [batch_size, number_of_classes], "
            f"but received {tuple(logits.shape)}."
        )

    if ground_truth_class_indices.ndim != 1:
        raise ValueError(
            "'ground_truth_class_indices' must have shape [batch_size], "
            f"but received {tuple(ground_truth_class_indices.shape)}."
        )

    if logits.size(0) != ground_truth_class_indices.size(0):
        raise ValueError(
            f"Batch size mismatch: logits batch={logits.size(0)}, "
            f"ground_truth batch={ground_truth_class_indices.size(0)}."
        )


def _convert_logits_to_predicted_class_indices(logits: Tensor) -> Tensor:
    """
    Converts class logits [batch_size, number_of_classes] into predicted class
    indices via an argmax operation.
    """
    return logits.argmax(dim=1)


def _compute_correct_prediction_mask(
    predicted_class_indices: Tensor,
    ground_truth_class_indices: Tensor,
) -> Tensor:
    """
    Computes a boolean mask indicating which predictions are correct.
    """
    return predicted_class_indices.eq(ground_truth_class_indices)


def _compute_mean_classification_accuracy(
    correct_prediction_mask: Tensor,
) -> float:
    """
    Computes the mean accuracy over the batch as a Python float.
    """
    return correct_prediction_mask.float().mean().item()
