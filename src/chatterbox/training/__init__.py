"""
VoicefyTTS Fine-Tuning Toolkit

Complete toolkit for fine-tuning VoicefyTTS on custom voices.

Usage:
    # Preprocess dataset
    python -m chatterbox.training.cli preprocess --input ./raw --output ./processed

    # Train model
    python -m chatterbox.training.cli train --dataset ./processed --output ./checkpoints

    # Generate with fine-tuned model
    python -m chatterbox.training.cli generate --checkpoint ./checkpoints/final --text "Hello"
"""
from .config import TrainingConfig
from .dataset import VoicefyDataset
from .collate import VoicefyCollator
from .train import train
from .preprocessing import AudioPreprocessor, SileroVAD, preprocess_dataset

__all__ = [
    "TrainingConfig",
    "VoicefyDataset",
    "VoicefyCollator",
    "train",
    "AudioPreprocessor",
    "SileroVAD",
    "preprocess_dataset"
]
