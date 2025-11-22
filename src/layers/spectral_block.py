from typing import Optional

from torch import Tensor
import torch.nn as nn

from .spectral_memory_operator import SpectralMemoryOperator


class SpectralBlock(nn.Module):
    """
    Dual-path block that combines spatial processing with spectral memory.

    The block decomposes the transformation into:
      - a spatial branch, which focuses on local spatial interactions via
        depthwise and pointwise convolutions, and
      - a spectral-memory branch, which applies the SpectralMemoryOperator
        in the frequency domain and projects back to the spatial domain.

    A gating tensor is derived from the spectral branch and used to modulate
    the contribution of the spectral path when fusing the two branches.
    The final output is obtained by adding a residual connection to the fused
    representation.
    """

    DEFAULT_KERNEL_SIZE: int = 3
    DEFAULT_PADDING: int = 1
    DEPTHWISE_GROUPS_FACTOR: int = 1  # groups = channels * factor

    def __init__(
        self,
        channels: int,
        memory_length: int = 4,
        use_learned_alpha: bool = True,
        gating_hidden_dimension: Optional[int] = None,
        gating_scale: float = 0.3,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.memory_length = memory_length
        self.use_learned_alpha = use_learned_alpha
        self.gating_scale = gating_scale
        self.gating_hidden_dimension = gating_hidden_dimension

        (
            self.spatial_normalization,
            self.spatial_depthwise_convolution,
            self.spatial_pointwise_convolution,
            self.spatial_activation,
        ) = self._build_spatial_branch_modules(channels)

        (
            self.spectral_normalization,
            self.spectral_input_projection,
            self.spectral_memory_operator,
            self.spectral_output_projection,
        ) = self._build_spectral_branch_modules(
            channels=channels,
            memory_length=memory_length,
            use_learned_alpha=use_learned_alpha,
            gating_hidden_dimension=gating_hidden_dimension,
            gating_scale=gating_scale,
        )

        self.spectral_to_gating_module = self._build_spectral_gating_module(channels)

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    def _build_spatial_branch_modules(
        self,
        channels: int,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        """
        Builds the spatial branch consisting of:
            - BatchNorm2d
            - depthwise convolution
            - pointwise convolution
            - non-linear activation
        """
        spatial_normalization = nn.BatchNorm2d(channels)

        depthwise_convolution = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.DEFAULT_KERNEL_SIZE,
            padding=self.DEFAULT_PADDING,
            groups=channels * self.DEPTHWISE_GROUPS_FACTOR,
        )

        pointwise_convolution = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )

        activation = nn.SiLU(inplace=True)

        return (
            spatial_normalization,
            depthwise_convolution,
            pointwise_convolution,
            activation,
        )

    def _build_spectral_branch_modules(
        self,
        channels: int,
        memory_length: int,
        use_learned_alpha: bool,
        gating_hidden_dimension: Optional[int],
        gating_scale: float,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        """
        Builds the spectral-memory branch consisting of:
            - BatchNorm2d
            - 1×1 convolution (input projection)
            - SpectralMemoryOperator
            - 1×1 convolution (output projection)
        """
        spectral_normalization = nn.BatchNorm2d(channels)

        spectral_input_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )

        spectral_memory_operator = SpectralMemoryOperator(
            channels=channels,
            memory_length=memory_length,
            use_learned_alpha=use_learned_alpha,
            gating_hidden_dimension=gating_hidden_dimension,
            gating_scale=gating_scale,
        )

        spectral_output_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )

        return (
            spectral_normalization,
            spectral_input_projection,
            spectral_memory_operator,
            spectral_output_projection,
        )

    def _build_spectral_gating_module(self, channels: int) -> nn.Module:
        """
        Builds a gating module that converts spectral-branch features into
        a spatially broadcasted per-channel gate in [0, 1].
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    # -------------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------------

    def _compute_spatial_branch_representation(self, input_tensor: Tensor) -> Tensor:
        """
        Applies the spatial branch to the input tensor.

        The spatial branch focuses on local spatial interactions via depthwise
        and pointwise convolutions.
        """
        normalized_tensor = self.spatial_normalization(input_tensor)
        spatial_representation = self.spatial_depthwise_convolution(normalized_tensor)
        spatial_representation = self.spatial_activation(spatial_representation)
        spatial_representation = self.spatial_pointwise_convolution(
            spatial_representation
        )
        return spatial_representation

    def _compute_spectral_branch_representation(self, input_tensor: Tensor) -> Tensor:
        """
        Applies the spectral-memory branch to the input tensor.

        The spectral branch first normalizes the input, then projects it,
        applies the SpectralMemoryOperator, and finally projects back to the
        original channel dimension.
        """
        normalized_tensor = self.spectral_normalization(input_tensor)
        spectral_representation = self.spectral_input_projection(normalized_tensor)
        spectral_representation = self.spectral_memory_operator(spectral_representation)
        spectral_representation = self.spectral_output_projection(
            spectral_representation
        )
        return spectral_representation

    def _compute_spectral_gating_tensor(self, spectral_representation: Tensor) -> Tensor:
        """
        Computes a gating tensor from the spectral branch representation.

        The gating tensor has shape [B, C, 1, 1] and is broadcastable to the
        spatial dimensions of the feature maps.
        """
        return self.spectral_to_gating_module(spectral_representation)

    def _fuse_spatial_and_spectral_representations(
        self,
        spatial_representation: Tensor,
        spectral_representation: Tensor,
        spectral_gating_tensor: Tensor,
    ) -> Tensor:
        """
        Fuses the spatial and spectral branches by modulating the spectral
        representation with a gating tensor and adding it to the spatial branch.
        """
        gated_spectral_representation = spectral_gating_tensor * spectral_representation
        fused_representation = spatial_representation + gated_spectral_representation
        return fused_representation

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Applies the dual-path spectral memory block to the input tensor.

        Args:
            input_tensor:
                Tensor of shape [batch_size, channels, height, width].

        Returns:
            Tensor of shape [batch_size, channels, height, width] obtained
            by adding a residual connection to the fused spatial–spectral
            representation.
        """
        residual_tensor = input_tensor

        spatial_representation = self._compute_spatial_branch_representation(
            input_tensor
        )
        spectral_representation = self._compute_spectral_branch_representation(
            input_tensor
        )
        spectral_gating_tensor = self._compute_spectral_gating_tensor(
            spectral_representation
        )

        fused_representation = self._fuse_spatial_and_spectral_representations(
            spatial_representation=spatial_representation,
            spectral_representation=spectral_representation,
            spectral_gating_tensor=spectral_gating_tensor,
        )

        output_tensor = residual_tensor + fused_representation
        return output_tensor
