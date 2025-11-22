from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import Tensor
import torch.nn as nn

from .spectral_block import SpectralBlock


class SpectralMemoryUNetDecoder(nn.Module):
    """
    Decoder component for a U-Net style architecture using spectral-memory blocks.

    The decoder receives:
      - the deepest encoder representation, and
      - the list of skip connection feature maps from the encoder.

    It progressively upsamples the representation and fuzionează cu skip-urile
    prin concatenare și blocuri de tip SpectralMemoryDualPathBlock.
    """

    def __init__(
        self,
        output_channels: int,
        encoder_channels_per_stage: Sequence[int],
        blocks_per_stage: Union[int, Sequence[int]] = 2,
        block_class: Type[nn.Module] = SpectralBlock,
        upsampling_mode: str = "bilinear",
        use_align_corners_for_upsampling: bool = False,
        memory_length: int = 4,
        use_learned_alpha: bool = True,
        gating_hidden_dimension: Optional[int] = None,
        gating_scale: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder_channels_per_stage = list(encoder_channels_per_stage)
        self.block_class = block_class
        self.upsampling_mode = upsampling_mode
        self.use_align_corners_for_upsampling = use_align_corners_for_upsampling

        self.blocks_per_stage = self._resolve_blocks_per_stage(
            blocks_per_stage=blocks_per_stage,
            number_of_stages=len(self.encoder_channels_per_stage) - 1,
        )

        (
            self.decoder_stages,
            self.upsampling_layers,
            self.skip_merging_convolutions,
        ) = self._build_decoder_stages(
            encoder_channels_per_stage=self.encoder_channels_per_stage,
            blocks_per_stage=self.blocks_per_stage,
            block_class=block_class,
            memory_length=memory_length,
            use_learned_alpha=use_learned_alpha,
            gating_hidden_dimension=gating_hidden_dimension,
            gating_scale=gating_scale,
        )

        self.output_projection = nn.Conv2d(
            in_channels=self.encoder_channels_per_stage[0],
            out_channels=output_channels,
            kernel_size=1,
        )

    # -------------------------------------------------------------------------
    # Configuration helpers
    # -------------------------------------------------------------------------

    def _resolve_blocks_per_stage(
        self,
        blocks_per_stage: Union[int, Sequence[int]],
        number_of_stages: int,
    ) -> List[int]:
        if isinstance(blocks_per_stage, int):
            return [blocks_per_stage] * number_of_stages

        if len(blocks_per_stage) != number_of_stages:
            raise ValueError(
                "The length of 'blocks_per_stage' must match the number of decoder stages. "
                f"Got {len(blocks_per_stage)} vs {number_of_stages}."
            )
        return list(blocks_per_stage)

    def _build_decoder_stages(
        self,
        encoder_channels_per_stage: Sequence[int],
        blocks_per_stage: Sequence[int],
        block_class: Type[nn.Module],
        memory_length: int,
        use_learned_alpha: bool,
        gating_hidden_dimension: Optional[int],
        gating_scale: float,
    ) -> Tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList]:
        """
        Builds the decoder stages, upsampling layers, and skip-merging convolutions.
        """
        number_of_stages = len(encoder_channels_per_stage) - 1

        decoder_stages = nn.ModuleList()
        upsampling_layers = nn.ModuleList()
        skip_merging_convolutions = nn.ModuleList()

        for stage_index in range(number_of_stages):
            # We go from deepest to shallower stages:
            index_from_bottom = number_of_stages - 1 - stage_index

            input_channels_from_bottom = encoder_channels_per_stage[index_from_bottom]
            skip_channels = encoder_channels_per_stage[index_from_bottom - 1]
            output_channels = skip_channels

            upsampling_layer = self._build_upsampling_layer(
                input_channels=input_channels_from_bottom,
                output_channels=output_channels,
            )
            upsampling_layers.append(upsampling_layer)

            skip_merging_convolution = nn.Conv2d(
                in_channels=output_channels + skip_channels,
                out_channels=output_channels,
                kernel_size=1,
            )
            skip_merging_convolutions.append(skip_merging_convolution)

            stage = self._build_single_decoder_stage(
                channels=output_channels,
                number_of_blocks=blocks_per_stage[stage_index],
                block_class=block_class,
                memory_length=memory_length,
                use_learned_alpha=use_learned_alpha,
                gating_hidden_dimension=gating_hidden_dimension,
                gating_scale=gating_scale,
            )
            decoder_stages.append(stage)

        return decoder_stages, upsampling_layers, skip_merging_convolutions

    def _build_upsampling_layer(
        self,
        input_channels: int,
        output_channels: int,
    ) -> nn.Module:
        """
        Builds an upsampling layer between decoder stages.
        """
        if self.upsampling_mode == "transposed_convolution":
            return nn.ConvTranspose2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=2,
                stride=2,
            )

        # Interpolation-based upsampling, followed by a 1x1 conv if needed:
        if input_channels != output_channels:
            return nn.Sequential(
                nn.Upsample(
                    scale_factor=2.0,
                    mode=self.upsampling_mode,
                    align_corners=self.use_align_corners_for_upsampling
                    if self.upsampling_mode in ("bilinear", "bicubic")
                    else None,
                ),
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1,
                ),
            )

        return nn.Upsample(
            scale_factor=2.0,
            mode=self.upsampling_mode,
            align_corners=self.use_align_corners_for_upsampling
            if self.upsampling_mode in ("bilinear", "bicubic")
            else None,
        )

    def _build_single_decoder_stage(
        self,
        channels: int,
        number_of_blocks: int,
        block_class: Type[nn.Module],
        memory_length: int,
        use_learned_alpha: bool,
        gating_hidden_dimension: Optional[int],
        gating_scale: float,
    ) -> nn.Sequential:
        """
        Builds a single decoder stage with several spectral-memory blocks.
        """
        blocks: List[nn.Module] = []

        for block_index in range(number_of_blocks):
            block = block_class(
                channels=channels,
                memory_length=memory_length,
                use_learned_alpha=use_learned_alpha,
                gating_hidden_dimension=gating_hidden_dimension,
                gating_scale=gating_scale,
            )
            blocks.append(block)

        return nn.Sequential(*blocks)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        deepest_encoder_representation: Tensor,
        encoder_skip_feature_maps: Sequence[Tensor],
    ) -> Tensor:
        """
        Forward pass through the decoder.

        Args:
            deepest_encoder_representation:
                Tensor from the deepest encoder stage.
            encoder_skip_feature_maps:
                Sequence of feature maps from the encoder stages, ordered
                from shallowest to deepest.

        Returns:
            Tensor of shape [batch_size, output_channels, height, width].
        """
        x = deepest_encoder_representation
        number_of_decoder_stages = len(self.decoder_stages)

        for stage_index in range(number_of_decoder_stages):
            upsampling_layer = self.upsampling_layers[stage_index]
            skip_merging_convolution = self.skip_merging_convolutions[stage_index]
            decoder_stage = self.decoder_stages[stage_index]

            x = upsampling_layer(x)

            # Determine which skip to use: we traverse encoder_skip_feature_maps from the end.
            skip_index = len(encoder_skip_feature_maps) - 2 - stage_index
            skip_tensor = encoder_skip_feature_maps[skip_index]

            x = torch.cat([x, skip_tensor], dim=1)
            x = skip_merging_convolution(x)
            x = decoder_stage(x)

        logits = self.output_projection(x)
        return logits
