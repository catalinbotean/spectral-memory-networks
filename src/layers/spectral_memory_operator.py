from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMemoryOperator(nn.Module):
    """
    Spectral Memory Operator (SMO)

    High-level steps:
      1. Compute 2D real FFT on the feature map.
      2. Maintain a temporal queue of batch-averaged spectra.
      3. Aggregate memory with learnable alpha weights.
      4. Modulate memory using a content-aware gating MLP.
      5. Apply inverse FFT and fuse with a residual connection.
    """

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

        if self.memory_length <= 0:
            raise ValueError("memory_length must be positive.")

        if use_learned_alpha:
            initial_alpha = torch.linspace(1.0, -1.0, memory_length)
            self.alpha_weights = nn.Parameter(initial_alpha)
        else:
            self.register_buffer(
                "alpha_buffer",
                torch.ones(memory_length) / memory_length,
            )

        if gating_hidden_dimension is None:
            gating_hidden_dimension = max(32, channels // 2)

        self.gating_mlp = nn.Sequential(
            nn.Linear(channels, gating_hidden_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(gating_hidden_dimension, channels),
        )

        self.real_memory_scale = nn.Parameter(torch.ones(1))
        self.imaginary_memory_scale = nn.Parameter(torch.ones(1))

        self.memory_real: Optional[List[torch.Tensor]] = None
        self.memory_imaginary: Optional[List[torch.Tensor]] = None

    def _ensure_memory_initialized(self, real_part: torch.Tensor) -> None:
        """
        Initialize or reset the memory if the spectral shape has changed.
        """
        channels, height, width_frequency = real_part.shape

        if self.memory_real is None:
            self.memory_real = []
            self.memory_imaginary = []
            return

        if len(self.memory_real) == 0:
            return

        reference = self.memory_real[0]
        if reference.shape != (channels, height, width_frequency):
            self.memory_real = []
            self.memory_imaginary = []

    def _update_memory(
        self,
        real_mean: torch.Tensor,
        imaginary_mean: torch.Tensor,
    ) -> None:
        """
        Push the new batch-aggregated spectrum into the memory queue.
        """
        assert self.memory_real is not None
        assert self.memory_imaginary is not None

        self.memory_real.insert(0, real_mean.detach())
        self.memory_imaginary.insert(0, imaginary_mean.detach())

        if len(self.memory_real) > self.memory_length:
            self.memory_real.pop()
            self.memory_imaginary.pop()

    def _aggregate_memory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate the spectral memory using alpha weights.

        Returns:
            aggregated_real:  [C, H, Wf]
            aggregated_imag:  [C, H, Wf]
        """
        assert self.memory_real is not None
        assert self.memory_imaginary is not None

        current_memory_length = len(self.memory_real)
        if current_memory_length == 0:
            raise RuntimeError(
                "Attempted to aggregate spectral memory but the memory is empty."
            )

        memory_real_stack = torch.stack(self.memory_real, dim=0)
        memory_imag_stack = torch.stack(self.memory_imaginary, dim=0)

        if self.use_learned_alpha:
            alpha = torch.softmax(
                self.alpha_weights[:current_memory_length], dim=0
            )
        else:
            alpha = self.alpha_buffer[:current_memory_length]

        alpha = alpha.view(current_memory_length, 1, 1, 1)

        aggregated_real = (alpha * memory_real_stack).sum(dim=0)
        aggregated_imag = (alpha * memory_imag_stack).sum(dim=0)

        return aggregated_real, aggregated_imag

    def _compute_gating_tensor(
        self,
        input_tensor: torch.Tensor,
        output_height: int,
        output_width_frequency: int,
    ) -> torch.Tensor:
        """
        Compute the content-aware gating tensor based on global average pooled features.

        Returns:
            gating_tensor: [B, C, H, Wf]
        """
        batch_size, channels, _, _ = input_tensor.shape

        pooled = F.adaptive_avg_pool2d(input_tensor, output_size=1).view(
            batch_size, channels
        )

        gating_values = self.gating_mlp(pooled)
        gating_values = torch.tanh(gating_values) * self.gating_scale
        gating_values = gating_values.view(batch_size, channels, 1, 1)

        gating_tensor = gating_values.expand(
            -1, -1, output_height, output_width_frequency
        )
        return gating_tensor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: [B, C, H, W]

        Returns:
            output_tensor: [B, C, H, W] with spectral memory enhancement.
        """
        batch_size, channels, height, width = input_tensor.shape

        frequency_tensor = torch.fft.rfft2(input_tensor, norm="ortho")
        real_part = frequency_tensor.real
        imaginary_part = frequency_tensor.imag
        width_frequency = real_part.shape[-1]

        real_mean = real_part.mean(dim=0)
        imaginary_mean = imaginary_part.mean(dim=0)

        self._ensure_memory_initialized(real_mean)
        assert self.memory_real is not None
        assert self.memory_imaginary is not None

        self._update_memory(real_mean, imaginary_mean)

        aggregated_real, aggregated_imag = self._aggregate_memory()

        aggregated_real_batch = aggregated_real.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        aggregated_imag_batch = aggregated_imag.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )

        gating_tensor = self._compute_gating_tensor(
            input_tensor=input_tensor,
            output_height=height,
            output_width_frequency=width_frequency,
        )

        gated_real = gating_tensor * aggregated_real_batch
        gated_imag = gating_tensor * aggregated_imag_batch

        gated_real = self.real_memory_scale * gated_real
        gated_imag = self.imaginary_memory_scale * gated_imag

        fused_real = real_part + gated_real
        fused_imag = imaginary_part + gated_imag

        fused_complex = torch.complex(fused_real, fused_imag)

        reconstructed_tensor = torch.fft.irfft2(
            fused_complex,
            s=(height, width),
            norm="ortho",
        )

        output_tensor = input_tensor + reconstructed_tensor
        return output_tensor
