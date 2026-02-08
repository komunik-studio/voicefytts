"""
Audio Effects for VoicefyTTS

Post-processing effects for speed and pitch control using torchaudio SoX effects.
"""

import torch
import torchaudio
from typing import Optional


class AudioEffects:
    """
    Post-processing audio effects for speed and pitch control.
    
    Uses torchaudio's SoX effects for high-quality audio processing:
    - Speed: tempo effect (maintains pitch)
    - Pitch: pitch + rate effects (maintains duration)
    
    Args:
        sample_rate: Audio sample rate in Hz. Default is 24000.
    
    Example:
        >>> effects = AudioEffects(sample_rate=24000)
        >>> wav = torch.randn(24000)  # 1 second of audio
        >>> 
        >>> # Make 50% faster
        >>> wav_fast = effects.change_speed(wav, speed=1.5)
        >>> 
        >>> # Shift pitch up by 6 semitones
        >>> wav_high = effects.change_pitch(wav, semitones=6)
        >>> 
        >>> # Apply both
        >>> wav_processed = effects.apply_effects(wav, speed=1.2, pitch=3)
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
    
    def change_speed(
        self,
        wav: torch.Tensor,
        speed: float = 1.0
    ) -> torch.Tensor:
        """
        Change playback speed without affecting pitch.
        
        Uses SoX tempo effect which maintains pitch while changing duration.
        
        Args:
            wav: Audio tensor [samples] or [batch, samples]
            speed: Speed factor (0.5 = 50% slower, 2.0 = 2x faster)
                   Clamped to range [0.5, 2.0]
        
        Returns:
            Speed-adjusted audio tensor with same shape as input
        
        Example:
            >>> wav = torch.randn(24000)  # 1 second
            >>> wav_fast = effects.change_speed(wav, speed=2.0)
            >>> wav_fast.shape[0]  # ~12000 (half duration)
        """
        if speed == 1.0:
            return wav
        
        # Clamp to safe range
        speed = max(0.5, min(2.0, speed))
        
        # Apply tempo effect
        effects = [["tempo", str(speed)]]
        
        # Handle batch dimension
        is_batched = wav.dim() > 1
        if not is_batched:
            wav = wav.unsqueeze(0)
        
        wav_out, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav,
            self.sample_rate,
            effects,
            channels_first=True
        )
        
        return wav_out if is_batched else wav_out.squeeze(0)
    
    def change_pitch(
        self,
        wav: torch.Tensor,
        semitones: float = 0.0
    ) -> torch.Tensor:
        """
        Change pitch without affecting speed/duration.
        
        Uses SoX pitch + rate effects to shift pitch while maintaining duration.
        
        Args:
            wav: Audio tensor [samples] or [batch, samples]
            semitones: Pitch shift in semitones (-12 to +12)
                       -12 = one octave down, +12 = one octave up
                       Clamped to range [-12, 12]
        
        Returns:
            Pitch-shifted audio tensor with approximately same duration
        
        Example:
            >>> wav = torch.randn(24000)  # 1 second
            >>> wav_high = effects.change_pitch(wav, semitones=6)
            >>> wav_high.shape[0]  # ~24000 (same duration)
        """
        if semitones == 0.0:
            return wav
        
        # Clamp to safe range
        semitones = max(-12, min(12, semitones))
        cents = int(semitones * 100)
        
        # Apply pitch + rate to maintain duration
        effects = [
            ["pitch", str(cents)],
            ["rate", str(self.sample_rate)],
        ]
        
        # Handle batch dimension
        is_batched = wav.dim() > 1
        if not is_batched:
            wav = wav.unsqueeze(0)
        
        wav_out, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav,
            self.sample_rate,
            effects,
            channels_first=True
        )
        
        return wav_out if is_batched else wav_out.squeeze(0)
    
    def apply_effects(
        self,
        wav: torch.Tensor,
        speed: float = 1.0,
        pitch: float = 0.0
    ) -> torch.Tensor:
        """
        Apply both speed and pitch effects.
        
        Effects are applied in order: speed first, then pitch.
        
        Args:
            wav: Audio tensor [samples] or [batch, samples]
            speed: Speed factor (0.5 - 2.0), default 1.0 (no change)
            pitch: Pitch shift in semitones (-12 to +12), default 0.0 (no change)
        
        Returns:
            Processed audio tensor
        
        Example:
            >>> # Make 20% slower and shift down 3 semitones
            >>> wav_processed = effects.apply_effects(wav, speed=0.8, pitch=-3)
        """
        # Apply speed first, then pitch
        if speed != 1.0:
            wav = self.change_speed(wav, speed)
        if pitch != 0.0:
            wav = self.change_pitch(wav, pitch)
        return wav
    
    def __repr__(self) -> str:
        return f"AudioEffects(sample_rate={self.sample_rate}Hz)"
