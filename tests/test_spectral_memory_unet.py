import torch
from src.models.spectral_memory_unet import SpectralMemoryUNet

def test_unet_forward():
    model = SpectralMemoryUNet(
        input_channels=3,
        output_channels=1,
        number_of_encoder_stages=4,
        base_number_of_channels=16,
        memory_length=2,
    )
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 1, 128, 128)
