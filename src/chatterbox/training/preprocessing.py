"""
Audio Preprocessing for VoicefyTTS Fine-Tuning

Provides audio preprocessing utilities:
- VAD (Voice Activity Detection) using Silero VAD
- Audio normalization
- Silence trimming
- Resampling
- Format conversion
"""

import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a segment of audio with timestamps."""
    start_ms: int
    end_ms: int
    audio: torch.Tensor
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD.

    Detects speech segments in audio files for preprocessing.
    """

    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250, min_silence_ms: int = 100):
        """
        Initialize Silero VAD.

        Args:
            threshold: Speech detection threshold (0-1). Higher = stricter.
            min_speech_ms: Minimum speech duration to keep (ms)
            min_silence_ms: Minimum silence duration to consider as pause (ms)
        """
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )

        self.get_speech_timestamps = self.utils[0]
        self.save_audio = self.utils[1]
        self.read_audio = self.utils[2]
        self.vad_sr = 16000  # Silero VAD requires 16kHz

        logger.info("Silero VAD loaded successfully")

    def detect_speech(
        self,
        audio: Union[torch.Tensor, str, Path],
        sample_rate: int = None
    ) -> List[dict]:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio tensor [samples] or path to audio file
            sample_rate: Sample rate (required if audio is tensor)

        Returns:
            List of dicts with 'start' and 'end' keys (in samples at 16kHz)
        """
        if isinstance(audio, (str, Path)):
            wav = self.read_audio(str(audio), sampling_rate=self.vad_sr)
        else:
            # Resample if needed
            if sample_rate != self.vad_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, self.vad_sr)
                wav = resampler(audio)
            else:
                wav = audio

        speech_timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            sampling_rate=self.vad_sr
        )

        return speech_timestamps

    def trim_silence(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        padding_ms: int = 100
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Trim leading and trailing silence from audio.

        Args:
            audio: Audio tensor [samples] or [1, samples]
            sample_rate: Sample rate
            padding_ms: Padding to add around speech (ms)

        Returns:
            Tuple of (trimmed_audio, start_sample, end_sample)
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        timestamps = self.detect_speech(audio, sample_rate)

        if not timestamps:
            logger.warning("No speech detected, returning original audio")
            return audio, 0, len(audio)

        # Convert padding to samples at VAD sample rate
        padding_samples_vad = int(padding_ms * self.vad_sr / 1000)

        # Get first and last speech
        start_vad = max(0, timestamps[0]['start'] - padding_samples_vad)
        end_vad = min(len(audio) * self.vad_sr // sample_rate, timestamps[-1]['end'] + padding_samples_vad)

        # Convert back to original sample rate
        start_sample = int(start_vad * sample_rate / self.vad_sr)
        end_sample = int(end_vad * sample_rate / self.vad_sr)

        return audio[start_sample:end_sample], start_sample, end_sample

    def split_on_silence(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        max_duration_s: float = 15.0,
        min_duration_s: float = 1.0,
        padding_ms: int = 100
    ) -> List[AudioSegment]:
        """
        Split audio into segments based on silence detection.

        Args:
            audio: Audio tensor [samples]
            sample_rate: Sample rate
            max_duration_s: Maximum segment duration (seconds)
            min_duration_s: Minimum segment duration (seconds)
            padding_ms: Padding around each segment (ms)

        Returns:
            List of AudioSegment objects
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        timestamps = self.detect_speech(audio, sample_rate)

        if not timestamps:
            return []

        segments = []
        padding_samples = int(padding_ms * sample_rate / 1000)

        for ts in timestamps:
            # Convert to original sample rate
            start = int(ts['start'] * sample_rate / self.vad_sr)
            end = int(ts['end'] * sample_rate / self.vad_sr)

            # Add padding
            start = max(0, start - padding_samples)
            end = min(len(audio), end + padding_samples)

            duration_s = (end - start) / sample_rate

            # Skip too short segments
            if duration_s < min_duration_s:
                continue

            # Split long segments
            if duration_s > max_duration_s:
                # Simple split at max duration
                chunk_samples = int(max_duration_s * sample_rate)
                for chunk_start in range(start, end, chunk_samples):
                    chunk_end = min(chunk_start + chunk_samples, end)
                    if (chunk_end - chunk_start) / sample_rate >= min_duration_s:
                        segments.append(AudioSegment(
                            start_ms=int(chunk_start * 1000 / sample_rate),
                            end_ms=int(chunk_end * 1000 / sample_rate),
                            audio=audio[chunk_start:chunk_end],
                            sample_rate=sample_rate
                        ))
            else:
                segments.append(AudioSegment(
                    start_ms=int(start * 1000 / sample_rate),
                    end_ms=int(end * 1000 / sample_rate),
                    audio=audio[start:end],
                    sample_rate=sample_rate
                ))

        return segments


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline for fine-tuning.
    """

    def __init__(
        self,
        target_sr: int = 24000,
        target_db: float = -20.0,
        vad_threshold: float = 0.5
    ):
        """
        Initialize preprocessor.

        Args:
            target_sr: Target sample rate (24000 for S3Gen)
            target_db: Target loudness in dB
            vad_threshold: VAD detection threshold
        """
        self.target_sr = target_sr
        self.target_db = target_db
        self.vad = SileroVAD(threshold=vad_threshold)

    def load_audio(self, path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """Load audio file and return tensor + sample rate."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        wav, sr = torchaudio.load(str(path))

        # Convert stereo to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        return wav.squeeze(0), sr

    def resample(self, audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if src_sr == dst_sr:
            return audio

        resampler = torchaudio.transforms.Resample(src_sr, dst_sr)
        return resampler(audio)

    def normalize_loudness(self, audio: torch.Tensor, target_db: float = None) -> torch.Tensor:
        """Normalize audio loudness to target dB."""
        target_db = target_db or self.target_db

        # Calculate current RMS
        rms = torch.sqrt(torch.mean(audio ** 2))
        current_db = 20 * torch.log10(rms + 1e-8)

        # Calculate gain
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain with clipping prevention
        normalized = audio * gain_linear
        max_val = torch.abs(normalized).max()
        if max_val > 0.99:
            normalized = normalized * (0.99 / max_val)

        return normalized

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        trim_silence: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Process a single audio file.

        Args:
            input_path: Path to input audio
            output_path: Optional path to save processed audio
            trim_silence: Whether to trim leading/trailing silence
            normalize: Whether to normalize loudness

        Returns:
            Processed audio tensor
        """
        # Load
        audio, sr = self.load_audio(input_path)

        # Resample
        audio = self.resample(audio, sr, self.target_sr)

        # Trim silence
        if trim_silence:
            audio, _, _ = self.vad.trim_silence(audio, self.target_sr)

        # Normalize
        if normalize:
            audio = self.normalize_loudness(audio)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), audio.unsqueeze(0), self.target_sr)

        return audio

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg'],
        trim_silence: bool = True,
        normalize: bool = True
    ) -> List[Path]:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: Audio file extensions to process
            trim_silence: Whether to trim silence
            normalize: Whether to normalize loudness

        Returns:
            List of processed file paths
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed = []

        for ext in extensions:
            for input_path in input_dir.glob(f"*{ext}"):
                output_path = output_dir / f"{input_path.stem}.wav"

                try:
                    self.process_file(input_path, output_path, trim_silence, normalize)
                    processed.append(output_path)
                    logger.info(f"Processed: {input_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {input_path.name}: {e}")

        logger.info(f"Processed {len(processed)} files")
        return processed


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    target_sr: int = 24000,
    trim_silence: bool = True,
    normalize: bool = True,
    target_db: float = -20.0
):
    """
    Preprocess an entire dataset for fine-tuning.

    Args:
        input_dir: Directory containing raw audio files
        output_dir: Output directory for processed files
        target_sr: Target sample rate
        trim_silence: Whether to trim silence
        normalize: Whether to normalize loudness
        target_db: Target loudness in dB
    """
    preprocessor = AudioPreprocessor(
        target_sr=target_sr,
        target_db=target_db
    )

    preprocessor.process_directory(
        input_dir,
        output_dir,
        trim_silence=trim_silence,
        normalize=normalize
    )
