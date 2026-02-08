"""
VoicefyTTS Fine-Tuning CLI

Command-line interface for fine-tuning VoicefyTTS models.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="VoicefyTTS Fine-Tuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess dataset
  python -m chatterbox.training.cli preprocess --input ./raw_audio --output ./processed

  # Train model
  python -m chatterbox.training.cli train --dataset ./processed --output ./checkpoints

  # Generate with fine-tuned model
  python -m chatterbox.training.cli generate --checkpoint ./checkpoints/final --text "Hello world"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess audio dataset")
    preprocess_parser.add_argument("--input", "-i", required=True, help="Input directory with raw audio")
    preprocess_parser.add_argument("--output", "-o", required=True, help="Output directory for processed audio")
    preprocess_parser.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate (default: 24000)")
    preprocess_parser.add_argument("--no-trim", action="store_true", help="Skip silence trimming")
    preprocess_parser.add_argument("--no-normalize", action="store_true", help="Skip loudness normalization")
    preprocess_parser.add_argument("--target-db", type=float, default=-20.0, help="Target loudness in dB (default: -20)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune model on dataset")
    train_parser.add_argument("--dataset", "-d", required=True, help="Path to processed dataset")
    train_parser.add_argument("--output", "-o", required=True, help="Output directory for checkpoints")
    train_parser.add_argument("--base-model", default="ResembleAI/chatterbox", help="Base model to fine-tune")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16", help="Mixed precision mode")
    train_parser.add_argument("--resume", help="Resume from checkpoint")
    train_parser.add_argument("--device", default="cuda", help="Device to use (cuda, cpu, mps)")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate speech with fine-tuned model")
    generate_parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint directory")
    generate_parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    generate_parser.add_argument("--output", "-o", default="output.wav", help="Output audio file")
    generate_parser.add_argument("--reference", "-r", help="Reference audio for voice cloning")
    generate_parser.add_argument("--preset", choices=["natural", "energetic", "calm", "narrator", "character"], help="Generation preset")
    generate_parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0)")
    generate_parser.add_argument("--pitch", type=float, default=0.0, help="Pitch shift in semitones (-12 to +12)")
    generate_parser.add_argument("--device", default="cuda", help="Device to use")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset format")
    validate_parser.add_argument("--dataset", "-d", required=True, help="Path to dataset")
    validate_parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")

    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_preprocess(args):
    """Preprocess audio dataset."""
    from .preprocessing import preprocess_dataset

    print(f"Preprocessing audio from {args.input}")
    print(f"  Target sample rate: {args.sample_rate} Hz")
    print(f"  Trim silence: {not args.no_trim}")
    print(f"  Normalize loudness: {not args.no_normalize}")
    print(f"  Target dB: {args.target_db}")

    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_sr=args.sample_rate,
        trim_silence=not args.no_trim,
        normalize=not args.no_normalize,
        target_db=args.target_db
    )

    print(f"Preprocessing complete. Output saved to {args.output}")


def cmd_train(args):
    """Train model on dataset."""
    from .config import TrainingConfig
    from .train import train

    config = TrainingConfig(
        train_data_path=args.dataset,
        output_dir=args.output,
        model_name=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume,
        device=args.device
    )

    print("Starting training with configuration:")
    print(f"  Dataset: {config.train_data_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")

    train(config)


def cmd_generate(args):
    """Generate speech with fine-tuned model."""
    import torch
    import torchaudio

    print(f"Loading checkpoint from {args.checkpoint}")

    # Load model
    from ..tts import ChatterboxTTS

    model = ChatterboxTTS.from_local(args.checkpoint, args.device)

    # Generate
    print(f"Generating speech for: '{args.text}'")

    wav = model.generate(
        text=args.text,
        audio_prompt_path=args.reference,
        preset=args.preset,
        speed=args.speed,
        pitch=args.pitch
    )

    # Save
    torchaudio.save(args.output, wav, model.sr)
    print(f"Audio saved to {args.output}")


def cmd_validate(args):
    """Validate dataset format."""
    import csv
    from pathlib import Path

    dataset_path = Path(args.dataset)
    metadata_path = dataset_path / "metadata.csv"
    wavs_dir = dataset_path / "wavs"

    issues = []

    # Check structure
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return

    if not metadata_path.exists():
        issues.append("Missing metadata.csv")
    else:
        # Validate metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for i, row in enumerate(reader, 1):
                if len(row) < 2:
                    issues.append(f"Line {i}: Invalid format (expected id|text)")
                    continue

                file_id = row[0]
                wav_path = wavs_dir / f"{file_id}.wav"

                if not wav_path.exists():
                    issues.append(f"Line {i}: Missing audio file: {wav_path.name}")

    if not wavs_dir.exists():
        issues.append("Missing wavs/ directory")
    else:
        # Check for orphan audio files
        audio_files = set(f.stem for f in wavs_dir.glob("*.wav"))
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                referenced = set(row[0] for row in reader if row)
            orphans = audio_files - referenced
            if orphans:
                issues.append(f"Orphan audio files (not in metadata): {', '.join(list(orphans)[:5])}...")

    # Report
    if issues:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")

        if args.fix:
            print("\nAttempting fixes...")
            # TODO: Implement auto-fixes
            print("Auto-fix not yet implemented")
    else:
        print("Dataset validation passed!")


if __name__ == "__main__":
    main()
