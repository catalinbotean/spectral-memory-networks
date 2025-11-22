from typing import Optional

import torch
import torch.nn as nn

from .spectral_memory_operator import SpectralMemoryOperator


class SpectralMemoryResidualBlock(nn.Module):
    """
    Residual block with optional Spectral Memory Operator (SMO).

    Structure:
        input -> Conv-BN-ReLU -> (SMO or Identity) -> Conv-BN-ReLU -> + skip
    """

    DEFAULT_KERNEL_SIZE: int = 3
    DEFAULT_PADDING: int = 1
    PROJECTION_KERNEL_SIZE: int = 1

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        use_spectral_memory: bool = True,
        memory_length: int = 4,
        use_learned_alpha: bool = True,
        gating_scale: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.input_projection = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=self.DEFAULT_KERNEL_SIZE,
                padding=self.DEFAULT_PADDING,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        if use_spectral_memory:
            self.spectral_memory_operator = SpectralMemoryOperator(
                channels=output_channels,
                memory_length=memory_length,
                use_learned_alpha=use_learned_alpha,
                gating_hidden_dimension=None,
                gating_scale=gating_scale,
            )
        else:
            self.spectral_memory_operator = nn.Identity()

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=self.DEFAULT_KERNEL_SIZE,
                padding=self.DEFAULT_PADDING,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.skip_projection: Optional[nn.Module]
        if input_channels != output_channels:
            self.skip_projection = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=self.PROJECTION_KERNEL_SIZE,
            )
        else:
            self.skip_projection = None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.skip_projection is None:
            skip_tensor = input_tensor
        else:
            skip_tensor = self.skip_projection(input_tensor)

        output_tensor = self.input_projection(input_tensor)
        output_tensor = self.spectral_memory_operator(output_tensor)
        output_tensor = self.output_projection(output_tensor)

        return output_tensor + skip_tensor
