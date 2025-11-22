from typing import List, Optional, Sequence, Tuple, Type, Union

from torch import Tensor
import torch.nn as nn

from .spectral_block import SpectralBlock


class SpectralMemoryUNetEncoder(nn.Module):
    """
    Encoder component for a U-Net style architecture using spectral-memory blocks.

    The encoder is organized in multiple stages. Each stage may contain several
    blocks (e.g., SpectralMemoryDualPathBlock) followed by a downsampling
    operation (max pooling or strided convolution).

    The encoder can optionally expose intermediate feature maps that may be used
    for auxiliary regularization losses (e.g., instance whitening, spectral
    regularization).
    """

    def __init(
        self,
        input_channels: int,
        number_of_stages: int = 4,
        base_number_of_channels: int = 64,
        channels_per_stage: Optional[Sequence[int]] = None,
        blocks_per_stage: Union[int, Sequence[int]] = 2,
        block_class: Type[nn.Module] = SpectralBlock,
        use_strided_convolution_for_downsampling: bool = False,
        downsampling_kernel_size: int = 2,
        downsampling_stride: int = 2,
        memory_length: int = 4,
        use_learned_alpha: bool = True,
        gating_hidden_dimension: Optional[int] = None,
        gating_scale: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.number_of_stages = number_of_stages
        self.block_class = block_class
        self.use_strided_convolution_for_downsampling = (
            use_strided_convolution_for_downsampling
        )
        self.downsampling_kernel_size = downsampling_kernel_size
        self.downsampling_stride = downsampling_stride

        (
            self.channels_per_stage,
            self.blocks_per_stage,
        ) = self._resolve_stage_configuration(
            number_of_stages=number_of_stages,
            base_number_of_channels=base_number_of_channels,
            channels_per_stage=channels_per_stage,
            blocks_per_stage=blocks_per_stage,
        )

        self.stem_convolution = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.channels_per_stage[0],
            kernel_size=3,
            padding=1,
        )

        (
            self.encoder_stages,
            self.downsampling_layers,
        ) = self._build_encoder_stages_and_downsamplers(
            channels_per_stage=self.channels_per_stage,
            blocks_per_stage=self.blocks_per_stage,
            block_class=block_class,
            memory_length=memory_length,
            use_learned_alpha=use_learned_alpha,
            gating_hidden_dimension=gating_hidden_dimension,
            gating_scale=gating_scale,
        )

    # -------------------------------------------------------------------------
    # Configuration helpers
    # -------------------------------------------------------------------------

    def _resolve_stage_configuration(
        self,
        number_of_stages: int,
        base_number_of_channels: int,
        channels_per_stage: Optional[Sequence[int]],
        blocks_per_stage: Union[int, Sequence[int]],
    ) -> Tuple[List[int], List[int]]:
        """
        Resolves the number of channels and blocks per stage.
        """
        if channels_per_stage is None:
            resolved_channels_per_stage = [
                base_number_of_channels * (2 ** stage_index)
                for stage_index in range(number_of_stages)
            ]
        else:
            if len(channels_per_stage) != number_of_stages:
                raise ValueError(
                    "The length of 'channels_per_stage' must be equal to 'number_of_stages'. "
                    f"Got {len(channels_per_stage)} vs {number_of_stages}."
                )
            resolved_channels_per_stage = list(channels_per_stage)

        if isinstance(blocks_per_stage, int):
            resolved_blocks_per_stage = [blocks_per_stage] * number_of_stages
        else:
            if len(blocks_per_stage) != number_of_stages:
                raise ValueError(
                    "The length of 'blocks_per_stage' must be equal to 'number_of_stages'. "
                    f"Got {len(blocks_per_stage)} vs {number_of_stages}."
                )
            resolved_blocks_per_stage = list(blocks_per_stage)

        return resolved_channels_per_stage, resolved_blocks_per_stage

    def _build_encoder_stages_and_downsamplers(
        self,
        channels_per_stage: Sequence[int],
        blocks_per_stage: Sequence[int],
        block_class: Type[nn.Module],
        memory_length: int,
        use_learned_alpha: bool,
        gating_hidden_dimension: Optional[int],
        gating_scale: float,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Builds the encoder stages and their corresponding downsampling operations.
        """
        stages = nn.ModuleList()
        downsampling_layers = nn.ModuleList()

        for stage_index, (stage_channels, number_of_blocks) in enumerate(
            zip(channels_per_stage, blocks_per_stage)
        ):
            if stage_index == 0:
                input_channels = stage_channels
            else:
                input_channels = channels_per_stage[stage_index - 1]

            stage = self._build_single_encoder_stage(
                input_channels=input_channels,
                output_channels=stage_channels,
                number_of_blocks=number_of_blocks,
                block_class=block_class,
                memory_length=memory_length,
                use_learned_alpha=use_learned_alpha,
                gating_hidden_dimension=gating_hidden_dimension,
                gating_scale=gating_scale,
            )
            stages.append(stage)

            if stage_index < len(channels_per_stage) - 1:
                downsampling_layer = self._build_downsampling_layer(
                    input_channels=stage_channels,
                    output_channels=channels_per_stage[stage_index + 1],
                )
                downsampling_layers.append(downsampling_layer)

        return stages, downsampling_layers

    def _build_single_encoder_stage(
        self,
        input_channels: int,
        output_channels: int,
        number_of_blocks: int,
        block_class: Type[nn.Module],
        memory_length: int,
        use_learned_alpha: bool,
        gating_hidden_dimension: Optional[int],
        gating_scale: float,
    ) -> nn.Sequential:
        """
        Builds a single encoder stage with several spectral-memory blocks.
        """
        blocks: List[nn.Module] = []

        for block_index in range(number_of_blocks):
            block = block_class(
                channels=output_channels,
                memory_length=memory_length,
                use_learned_alpha=use_learned_alpha,
                gating_hidden_dimension=gating_hidden_dimension,
                gating_scale=gating_scale,
            )
            blocks.append(block)

        return nn.Sequential(*blocks)

    def _build_downsampling_layer(
        self,
        input_channels: int,
        output_channels: int,
    ) -> nn.Module:
        """
        Builds a downsampling operation between encoder stages.
        """
        if self.use_strided_convolution_for_downsampling:
            return nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=self.downsampling_kernel_size,
                stride=self.downsampling_stride,
            )

        return nn.MaxPool2d(
            kernel_size=self.downsampling_kernel_size,
            stride=self.downsampling_stride,
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        input_tensor: Tensor,
        return_feature_maps_for_regularization: bool = False,
    ) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:
        """
        Forward pass through the encoder.

        Args:
            input_tensor:
                Tensor of shape [batch_size, input_channels, height, width].
            return_feature_maps_for_regularization:
                If True, returns a list of intermediate feature maps that
                can be used for auxiliary losses (e.g., instance whitening).

        Returns:
            If return_feature_maps_for_regularization is False:
                final_encoder_representation: Tensor
            If True:
                (final_encoder_representation, feature_maps_for_regularization)
        """
        feature_maps_for_skip_connections: List[Tensor] = []
        feature_maps_for_regularization: List[Tensor] = []

        x = self.stem_convolution(input_tensor)

        for stage_index, stage in enumerate(self.encoder_stages):
            x = stage(x)
            feature_maps_for_skip_connections.append(x)

            # Heuristic: collect feature maps from deeper stages for regularization
            if return_feature_maps_for_regularization and stage_index > 0:
                feature_maps_for_regularization.append(x)

            if stage_index < len(self.downsampling_layers):
                x = self.downsampling_layers[stage_index](x)

        self._last_skip_feature_maps = feature_maps_for_skip_connections

        if return_feature_maps_for_regularization:
            return x, feature_maps_for_regularization

        return x

    @property
    def skip_feature_maps(self) -> List[Tensor]:
        """
        Returns the feature maps from each encoder stage that are used as skip
        connections in the decoder.
        """
        if not hasattr(self, "_last_skip_feature_maps"):
            raise RuntimeError(
                "Skip feature maps are not available. "
                "Call the encoder's forward method before accessing them."
            )
        return self._last_skip_feature_maps
