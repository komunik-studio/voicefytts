"""
Training Configuration for VoicefyTTS
"""
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning VoicefyTTS models (T3/LLaMA)."""

    # Dataset
    train_data_path: str = "datasets/custom/train"
    val_data_path: str = "datasets/custom/val"
    dataset_format: str = "ljspeech"  # ljspeech | voicefy
    
    # Model
    model_name: str = "chatterbox-turbo"
    resume_from_checkpoint: Optional[str] = None
    freeze_voice_encoder: bool = True
    freeze_s3gen: bool = True
    
    # Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # no, fp16, bf16
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    
    # Output
    output_dir: str = "checkpoints"
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    
    def __post_init__(self):
        # Validation logic could go here
        pass
