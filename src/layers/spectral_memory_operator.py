from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SpectralMemoryOperator(nn.Module):
    """
    Spectral Memory Operator (SMO)

    This operator augments convolutional feature maps with a temporally aggregated
    memory of their frequency-domain representations.

    Conceptual pipeline:
        1. Transform input into frequency domain via 2D real FFT.
        2. Update and maintain a temporal queue of batch-averaged spectra.
        3. Aggregate stored spectra using learnable or fixed alpha weights.
        4. Generate a content-aware gating tensor using global context.
        5. Fuse current spectrum with gated spectral memory.
        6. Transform back to spatial domain using inverse FFT.
        7. Add residual connection in the spatial domain.

    Produces:
        output_tensor = input_tensor + IFFT( fused_frequency_representation )
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

        if memory_length <= 0:
            raise ValueError("memory_length must be positive.")

        # ------------------------------------------------------------------
        # Alpha weights for temporal aggregation
        # ------------------------------------------------------------------
        if use_learned_alpha:
            initial_alpha = torch.linspace(1.0, -1.0, memory_length)
            self.alpha_weights = nn.Parameter(initial_alpha)
        else:
            self.register_buffer(
                "alpha_buffer",
                torch.ones(memory_length) / memory_length,
            )

        # ------------------------------------------------------------------
        # Gating MLP
        # ------------------------------------------------------------------
        if gating_hidden_dimension is None:
            gating_hidden_dimension = max(32, channels // 2)

        self.gating_mlp = nn.Sequential(
            nn.Linear(channels, gating_hidden_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(gating_hidden_dimension, channels),
        )

        # Independent scaling of real and imaginary spectral contributions
        self.real_memory_scale = nn.Parameter(torch.ones(1))
        self.imaginary_memory_scale = nn.Parameter(torch.ones(1))

        # Spectral memory queues (store per-batch aggregated spectra)
        self.memory_real: Optional[List[Tensor]] = None
        self.memory_imaginary: Optional[List[Tensor]] = None

    # ==========================================================================
    # Memory initialization and management
    # ==========================================================================

    def _initialize_or_reset_memory_if_needed(self, spectral_real_slice: Tensor) -> None:
        """
        Ensures the internal memory queue matches the current spectral dimensions.

        If changed (e.g., different resolution), memory is reset.
        """
        current_shape = tuple(spectral_real_slice.shape)  # (C, H, Wf)

        if self.memory_real is None:
            self.memory_real = []
            self.memory_imaginary = []
            return

        if len(self.memory_real) == 0:
            return

        if self.memory_real[0].shape != current_shape:
            self.memory_real = []
            self.memory_imaginary = []

    def _update_temporal_spectral_memory(
        self,
        real_component: Tensor,
        imaginary_component: Tensor,
    ) -> None:
        """
        Inserts a new batch-averaged spectral slice at the front of the memory queue,
        respecting maximum memory length.
        """
        assert self.memory_real is not None
        assert self.memory_imaginary is not None

        self.memory_real.insert(0, real_component.detach())
        self.memory_imaginary.insert(0, imaginary_component.detach())

        if len(self.memory_real) > self.memory_length:
            self.memory_real.pop()
            self.memory_imaginary.pop()

    def _aggregate_temporal_memory(self) -> Tuple[Tensor, Tensor]:
        """
        Aggregates stored temporal spectral slices using alpha weighting.

        Returns:
            aggregated_real: tensor of shape [C, H, Wf]
            aggregated_imag: tensor of shape [C, H, Wf]
        """
        assert self.memory_real is not None
        assert self.memory_imaginary is not None

        if len(self.memory_real) == 0:
            raise RuntimeError("Spectral memory is empty.")

        memory_real_stack = torch.stack(self.memory_real, dim=0)
        memory_imag_stack = torch.stack(self.memory_imaginary, dim=0)

        # Alpha weight selection
        if self.use_learned_alpha:
            alpha = torch.softmax(
                self.alpha_weights[: len(self.memory_real)],
                dim=0,
            )
        else:
            alpha = self.alpha_buffer[: len(self.memory_real)]

        alpha = alpha.view(-1, 1, 1, 1)

        aggregated_real = (alpha * memory_real_stack).sum(dim=0)
        aggregated_imag = (alpha * memory_imag_stack).sum(dim=0)

        return aggregated_real, aggregated_imag

    # ==========================================================================
    # Gating mechanism
    # ==========================================================================

    def _compute_content_aware_gating_tensor(
        self,
        input_tensor: Tensor,
        height: int,
        width_frequency: int,
    ) -> Tensor:
        """
        Generates a spatially broadcasted gating tensor derived from
        globally pooled context features.
        """
        batch_size, channels, _, _ = input_tensor.shape

        pooled_features = F.adaptive_avg_pool2d(input_tensor, output_size=1).view(
            batch_size, channels
        )

        gating_vector = self.gating_mlp(pooled_features)
        gating_vector = torch.tanh(gating_vector) * self.gating_scale

        gating_tensor = gating_vector.view(batch_size, channels, 1, 1).expand(
            batch_size, channels, height, width_frequency
        )
        return gating_tensor

    # ==========================================================================
    # Frequency-domain fusion
    # ==========================================================================

    def _fuse_current_and_memory_spectra(
        self,
        real_part: Tensor,
        imag_part: Tensor,
        aggregated_real: Tensor,
        aggregated_imag: Tensor,
        gating_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Combines (current spectral components) + (gated memory components).
        """
        gated_real = gating_tensor * aggregated_real
        gated_imag = gating_tensor * aggregated_imag

        gated_real = self.real_memory_scale * gated_real
        gated_imag = self.imaginary_memory_scale * gated_imag

        fused_real = real_part + gated_real
        fused_imag = imag_part + gated_imag

        return fused_real, fused_imag

    # ==========================================================================
    # Forward Pass
    # ==========================================================================

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Enhances the input tensor via spectral memory fusion.

        Args:
            input_tensor: [batch, channels, height, width]

        Returns:
            output_tensor of same shape
        """
        batch_size, channels, height, width = input_tensor.shape

        # ---- 1. FFT ----
        frequency_tensor = torch.fft.rfft2(input_tensor, norm="ortho")
        real_part, imag_part = frequency_tensor.real, frequency_tensor.imag
        width_frequency = real_part.shape[-1]

        # ---- 2. Batch-mean spectrum and memory update ----
        real_mean = real_part.mean(dim=0)
        imag_mean = imag_part.mean(dim=0)

        self._initialize_or_reset_memory_if_needed(real_mean)
        self._update_temporal_spectral_memory(real_mean, imag_mean)

        # ---- 3. Memory aggregation ----
        aggregated_real, aggregated_imag = self._aggregate_temporal_memory()

        aggregated_real_batch = aggregated_real.unsqueeze(0).expand(batch_size, -1, -1, -1)
        aggregated_imag_batch = aggregated_imag.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # ---- 4. Content-aware gating ----
        gating_tensor = self._compute_content_aware_gating_tensor(
            input_tensor=input_tensor,
            height=height,
            width_frequency=width_frequency,
        )

        # ---- 5. Spectral fusion ----
        fused_real, fused_imag = self._fuse_current_and_memory_spectra(
            real_part=real_part,
            imag_part=imag_part,
            aggregated_real=aggregated_real_batch,
            aggregated_imag=aggregated_imag_batch,
            gating_tensor=gating_tensor,
        )

        # ---- 6. Inverse FFT ----
        fused_complex = torch.complex(fused_real, fused_imag)
        reconstructed_tensor = torch.fft.irfft2(
            fused_complex,
            s=(height, width),
            norm="ortho",
        )

        # ---- 7. Spatial-domain residual addition ----
        return input_tensor + reconstructed_tensor
