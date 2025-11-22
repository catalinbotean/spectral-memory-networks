from typing import Optional

from torch import Tensor
import torch.nn as nn

from .spectral_block import SpectralBlock


class SpectralMemoryResidualBlock(nn.Module):
    """
    Wrapper block that ensures input and output channel compatibility
    before applying a SpectralMemoryDualPathBlock.

    Note:
        The dual-path block already includes its own residual connection.
        Therefore, this wrapper does NOT add another residual link.
    """

    PROJECTION_KERNEL_SIZE: int = 1

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        memory_length: int = 4,
        use_learned_alpha: bool = True,
        gating_hidden_dimension: Optional[int] = None,
        gating_scale: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.channel_adapter = self._build_channel_adapter(
            input_channels,
            output_channels
        )

        self.spectral_memory_block = SpectralBlock(
            channels=output_channels,
            memory_length=memory_length,
            use_learned_alpha=use_learned_alpha,
            gating_hidden_dimension=gating_hidden_dimension,
            gating_scale=gating_scale,
        )

    # -------------------------------------------------------------------------

    def _build_channel_adapter(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        if in_channels == out_channels:
            return nn.Identity()

        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.PROJECTION_KERNEL_SIZE,
        )

    # -------------------------------------------------------------------------

    def forward(self, input_tensor: Tensor) -> Tensor:
        adapted_input = self.channel_adapter(input_tensor)
        return self.spectral_memory_block(adapted_input)
