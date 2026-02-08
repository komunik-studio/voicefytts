# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.
# Adapted for VoicefyTTS integration

"""
BigVGAN Vocoder Wrapper for VoicefyTTS

Integrates NVIDIA's BigVGAN v2 vocoder as a drop-in replacement for HiFi-GAN.
Provides improved quality (PESQ 4.36+ vs 4.0) and faster inference (~5760 kHz vs ~1000 kHz).
"""

import torch
import torch.nn as nn
from typing import Optional


class BigVGANVocoder(nn.Module):
    """
    Wrapper for BigVGAN v2 vocoder with HuggingFace integration.
    
    Presets:
        - 'quality': nvidia/bigvgan_v2_24khz_100band_256x (best quality, recommended)
        - 'fast': nvidia/bigvgan_base_24khz_100band (faster inference)
        - 'hifi': nvidia/bigvgan_v2_44khz_128band_512x (44.1kHz output)
    
    Args:
        preset: Model preset name. Default is 'quality'.
        use_cuda_kernel: Enable optimized CUDA kernels for faster inference. Default is True.
        device: Device to load model on. Default is 'cuda'.
    
    Example:
        >>> vocoder = BigVGANVocoder(preset='quality', device='cuda')
        >>> mel = torch.randn(1, 100, 256)  # [batch, n_mels, frames]
        >>> wav = vocoder(mel)  # [batch, 1, samples]
    """
    
    PRESETS = {
        'quality': 'nvidia/bigvgan_v2_24khz_100band_256x',
        'fast': 'nvidia/bigvgan_base_24khz_100band',
        'hifi': 'nvidia/bigvgan_v2_44khz_128band_512x',
    }
    
    def __init__(
        self,
        preset: str = 'quality',
        use_cuda_kernel: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        if preset not in self.PRESETS:
            raise ValueError(
                f"Invalid preset '{preset}'. "
                f"Available presets: {list(self.PRESETS.keys())}"
            )
        
        self.preset = preset
        self.device = device
        self.use_cuda_kernel = use_cuda_kernel and device == 'cuda'
        
        # Import BigVGAN (lazy import to avoid dependency issues)
        try:
            from bigvgan import BigVGAN
        except ImportError:
            raise ImportError(
                "BigVGAN is not installed. Install it with: pip install bigvgan"
            )
        
        # Load model from HuggingFace
        model_id = self.PRESETS[preset]
        print(f"[BigVGAN] Loading {preset} model from {model_id}...")
        
        self.model = BigVGAN.from_pretrained(
            model_id,
            use_cuda_kernel=self.use_cuda_kernel
        )
        
        # Move to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Remove weight norm for inference (improves speed)
        self.model.remove_weight_norm()
        
        print(f"[BigVGAN] Model loaded successfully on {device}")
        if self.use_cuda_kernel:
            print("[BigVGAN] Using optimized CUDA kernels for faster inference")
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform.
        
        Args:
            mel: Mel-spectrogram tensor [batch, n_mels, frames]
        
        Returns:
            Waveform tensor [batch, 1, samples]
        """
        with torch.no_grad():
            wav = self.model(mel)
        return wav
    
    @property
    def sample_rate(self) -> int:
        """Get output sample rate based on preset."""
        if self.preset == 'hifi':
            return 44100
        return 24000
    
    def __repr__(self) -> str:
        return (
            f"BigVGANVocoder("
            f"preset='{self.preset}', "
            f"use_cuda_kernel={self.use_cuda_kernel}, "
            f"device='{self.device}', "
            f"sample_rate={self.sample_rate}Hz)"
        )
