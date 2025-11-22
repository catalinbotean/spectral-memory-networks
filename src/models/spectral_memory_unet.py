from typing import List, Optional, Sequence, Tuple, Type, Union

from torch import Tensor
import torch.nn as nn

from src.layers import SpectralBlock, SpectralMemoryUNetDecoder, SpectralMemoryUNetEncoder


class SpectralMemoryUNet(nn.Module):
    """
    Full U-Net model composed of a spectral-memory encoder and decoder.

    The model can optionally return intermediate encoder feature maps that may
    be used to define auxiliary losses (e.g., instance whitening or spectral
    regularization at deeper layers).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        number_of_encoder_stages: int = 4,
        base_number_of_channels: int = 64,
        encoder_channels_per_stage: Optional[Sequence[int]] = None,
        encoder_blocks_per_stage: Union[int, Sequence[int]] = 2,
        decoder_blocks_per_stage: Union[int, Sequence[int]] = 2,
        block_class: Type[nn.Module] = SpectralBlock,
        use_strided_convolution_for_downsampling: bool = False,
        memory_length: int = 4,
        use_learned_alpha: bool = True,
        gating_hidden_dimension: Optional[int] = None,
        gating_scale: float = 0.3,
        upsampling_mode: str = "bilinear",
        use_align_corners_for_upsampling: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = SpectralMemoryUNetEncoder(
            input_channels=input_channels,
            number_of_stages=number_of_encoder_stages,
            base_number_of_channels=base_number_of_channels,
            channels_per_stage=encoder_channels_per_stage,
            blocks_per_stage=encoder_blocks_per_stage,
            block_class=block_class,
            use_strided_convolution_for_downsampling=use_strided_convolution_for_downsampling,
            memory_length=memory_length,
            use_learned_alpha=use_learned_alpha,
            gating_hidden_dimension=gating_hidden_dimension,
            gating_scale=gating_scale,
        )

        resolved_channels_per_stage = self.encoder.channels_per_stage

        self.decoder = SpectralMemoryUNetDecoder(
            output_channels=output_channels,
            encoder_channels_per_stage=resolved_channels_per_stage,
            blocks_per_stage=decoder_blocks_per_stage,
            block_class=block_class,
            upsampling_mode=upsampling_mode,
            use_align_corners_for_upsampling=use_align_corners_for_upsampling,
            memory_length=memory_length,
            use_learned_alpha=use_learned_alpha,
            gating_hidden_dimension=gating_hidden_dimension,
            gating_scale=gating_scale,
        )

    def forward(
        self,
        input_tensor: Tensor,
        return_encoder_features_for_regularization: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Args:
            input_tensor:
                Input image batch [batch_size, input_channels, height, width].
            return_encoder_features_for_regularization:
                If True, additionally returns a list of intermediate encoder
                feature maps that can be used for auxiliary losses.

        Returns:
            If return_encoder_features_for_regularization is False:
                segmentation_logits
            If True:
                (segmentation_logits, encoder_features_for_regularization)
        """
        if return_encoder_features_for_regularization:
            deepest_representation, regularization_features = self.encoder(
                input_tensor,
                return_feature_maps_for_regularization=True,
            )
        else:
            deepest_representation = self.encoder(
                input_tensor,
                return_feature_maps_for_regularization=False,
            )
            regularization_features = []

        skip_feature_maps = self.encoder.skip_feature_maps
        segmentation_logits = self.decoder(
            deepest_encoder_representation=deepest_representation,
            encoder_skip_feature_maps=skip_feature_maps,
        )

        if return_encoder_features_for_regularization:
            return segmentation_logits, regularization_features

        return segmentation_logits
