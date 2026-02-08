# VoicefyTTS

> **Fork of [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI**
>
> Enhanced TTS engine with BigVGAN v2 vocoder, speed/pitch controls, presets, tag normalization, and integrated fine-tuning toolkit.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🎯 What's Different from Original Chatterbox?

VoicefyTTS is a production-ready fork of Chatterbox with the following enhancements:

### ✅ Complete Changelog

| Feature | Original Chatterbox | VoicefyTTS |
|---------|---------------------|------------|
| **Watermark** | Perth (Resemble AI) | Voicefy placeholder (disabled by default) |
| **Vocoder** | HiFi-GAN only | ✅ **BigVGAN v2 (NVIDIA)** - PESQ 4.36+, ~5760 kHz with CUDA kernels |
| **Speed Control** | ❌ Not available | ✅ **0.5x - 2.0x** via torchaudio SoX effects |
| **Pitch Control** | ❌ Not available | ✅ **-12 to +12 semitones** |
| **Presets** | Manual tuning only | ✅ **5 optimized presets** (natural, energetic, calm, narrator, character) |
| **Tag Normalization** | Strict tags only | ✅ **20+ aliases** (e.g., [laughs] → [laugh]) |
| **Fine-tuning** | Separate repository | ✅ **Integrated toolkit** with VAD, preprocessing, CLI |
| **Package name** | `chatterbox-tts` | `voicefytts` |

### 📁 Files Changed

```
src/chatterbox/
├── __init__.py              # Updated version + exports
├── tts.py                   # +presets, +speed/pitch, +tags, -Perth watermark
├── tts_turbo.py             # +presets, +speed/pitch, +tags, -Perth watermark
├── vc.py                    # -Perth watermark
├── mtl_tts.py               # +presets, +speed/pitch, +tags, -Perth watermark
├── presets.py               # NEW: 5 generation presets
├── watermark/
│   ├── __init__.py          # NEW
│   └── voicefy_watermark.py # NEW: Voicefy watermark placeholder
├── tags/
│   ├── __init__.py          # NEW
│   └── normalizer.py        # NEW: Tag normalization with 20+ aliases
├── audio_effects/
│   ├── __init__.py          # NEW
│   └── effects.py           # NEW: Speed/pitch via torchaudio SoX
├── training/
│   ├── __init__.py          # NEW: Fine-tuning toolkit exports
│   ├── config.py            # NEW: Training configuration
│   ├── dataset.py           # NEW: LJSpeech dataset loader
│   ├── collate.py           # NEW: Batch collation
│   ├── train.py             # NEW: Training loop
│   ├── preprocessing.py     # NEW: VAD, silence trimming, normalization
│   └── cli.py               # NEW: Command-line interface
└── models/s3gen/
    ├── s3gen.py             # Updated: BigVGAN integration with fallback
    └── bigvgan_vocoder.py   # NEW: BigVGAN v2 wrapper
```

---

## Installation

```shell
pip install git+https://github.com/komunik-studio/voicefytts.git
```

Or install from source:

```shell
git clone https://github.com/komunik-studio/voicefytts.git
cd voicefytts
pip install -e .
```

### Dependencies Added

```toml
# pyproject.toml additions
bigvgan>=0.1.0  # NVIDIA BigVGAN v2 vocoder
```

---

## Usage

### Basic TTS

```python
import torchaudio as ta
from chatterbox import ChatterboxTTS

# Load model (uses BigVGAN v2 by default)
model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate with tags (aliases work automatically!)
text = "Hi there [laughs], how are you doing today?"  # [laughs] → [laugh]
wav = model.generate(text, audio_prompt_path="reference.wav")

ta.save("output.wav", wav, model.sr)
```

### Turbo Model (Faster)

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")
wav = model.generate("Hello world!", audio_prompt_path="reference.wav")
```

---

## 🆕 Generation Presets

VoicefyTTS includes optimized presets for common use cases:

```python
# Use predefined presets
wav = model.generate(text, preset="energetic")
```

| Preset | Description | Speed | Pitch | Exaggeration | CFG | Temperature |
|--------|-------------|-------|-------|--------------|-----|-------------|
| `natural` | General-purpose, balanced | 1.0x | 0 | 50% | 0.5 | 0.8 |
| `energetic` | Upbeat, advertisements | 1.1x | +1 | 70% | 0.4 | 0.9 |
| `calm` | Meditation, relaxation | 0.9x | -1 | 30% | 0.6 | 0.7 |
| `narrator` | Audiobooks, documentaries | 0.95x | 0 | 40% | 0.7 | 0.75 |
| `character` | Animation, games | 1.0x | 0 | 80% | 0.3 | 1.0 |

---

## 🆕 Manual Controls

Override preset values or manually tune all parameters:

```python
wav = model.generate(
    text="Hello world",

    # Preset (optional base)
    preset="calm",

    # Voice expressiveness
    exaggeration=0.5,      # 0.0-1.0 (higher = more expressive)
    cfg_weight=0.5,        # Classifier-free guidance (0.0-1.0)
    temperature=0.8,       # Sampling temperature (0.5-1.2)

    # Post-processing effects
    speed=0.8,             # 0.5-2.0 (tempo without pitch change)
    pitch=-2.0,            # -12 to +12 semitones
)
```

---

## Supported Tags & Aliases

VoicefyTTS supports 9 core paralinguistic tags with **20+ aliases**:

| Canonical Tag | Aliases |
|---------------|---------|
| `[laugh]` | `[laughs]`, `[laughter]`, `[haha]`, `[hehe]` |
| `[chuckle]` | `[chuckles]` |
| `[sigh]` | `[sighs]` |
| `[gasp]` | `[gasps]` |
| `[cough]` | `[coughs]`, `[coughing]` |
| `[clear throat]` | `[clears throat]`, `[ahem]` |
| `[sniff]` | `[sniffs]`, `[sniffing]` |
| `[groan]` | `[groans]`, `[groaning]`, `[ugh]` |
| `[shush]` | `[shh]`, `[shhh]` |

```python
# These all produce the same output:
"That's funny [laugh]"
"That's funny [laughs]"
"That's funny [haha]"
```

---

## 🛠️ Fine-Tuning Toolkit

VoicefyTTS includes a complete toolkit for fine-tuning on custom voices.

### CLI Commands

```bash
# Preprocess dataset (VAD, normalization, silence trimming)
python -m chatterbox.training.cli preprocess \
    --input ./raw_audio \
    --output ./processed

# Validate dataset format
python -m chatterbox.training.cli validate \
    --dataset ./processed

# Train model
python -m chatterbox.training.cli train \
    --dataset ./processed \
    --output ./checkpoints \
    --epochs 100 \
    --batch-size 4

# Generate with fine-tuned model
python -m chatterbox.training.cli generate \
    --checkpoint ./checkpoints/final \
    --text "Hello world" \
    --output output.wav
```

### Dataset Format (LJSpeech)

```
dataset/
├── wavs/
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ...
└── metadata.csv
```

**metadata.csv:**
```csv
audio001|Hello, this is the first sentence.
audio002|And this is the second sentence.
```

### Programmatic Usage

```python
from chatterbox.training import (
    TrainingConfig,
    AudioPreprocessor,
    preprocess_dataset,
    train
)

# Preprocess
preprocess_dataset(
    input_dir="./raw",
    output_dir="./processed",
    target_sr=24000,
    trim_silence=True,
    normalize=True
)

# Train
config = TrainingConfig(
    train_data_path="./processed",
    output_dir="./checkpoints",
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    device="cuda"
)
train(config)
```

---

## BigVGAN v2 Vocoder

VoicefyTTS uses NVIDIA's BigVGAN v2 as the default vocoder:

| Vocoder | Quality (PESQ) | Speed | GPU Memory |
|---------|----------------|-------|------------|
| HiFi-GAN (original) | 4.0 | ~1000 kHz | Low |
| **BigVGAN v2** | **4.36+** | **~5760 kHz** | Medium |

The vocoder is automatically selected:
- BigVGAN v2 is used by default when available
- Falls back to HiFi-GAN if BigVGAN fails to load

```python
# Force specific vocoder in S3Gen
from chatterbox.models.s3gen import S3Token2Wav

# BigVGAN (default)
s3gen = S3Token2Wav(vocoder_type='bigvgan', bigvgan_preset='quality')

# HiFi-GAN (fallback)
s3gen = S3Token2Wav(vocoder_type='hifigan')
```

BigVGAN presets:
- `quality`: Best quality (nvidia/bigvgan_v2_24khz_100band_256x)
- `fast`: Faster inference (nvidia/bigvgan_base_24khz_100band)
- `hifi`: 44.1kHz output (nvidia/bigvgan_v2_44khz_128band_512x)

---

## API Reference

### Main Classes

```python
from chatterbox import (
    ChatterboxTTS,           # Main TTS model
    ChatterboxVC,            # Voice conversion
    ChatterboxMultilingualTTS,  # Multilingual TTS
    SUPPORTED_LANGUAGES,     # Dict of supported languages
)

# VoicefyTTS-specific
from chatterbox import (
    get_preset,              # Get preset parameters
    list_presets,            # List available presets
    GENERATION_PRESETS,      # Dict of all presets
    normalize_tags,          # Normalize tag aliases
    validate_tags,           # Validate tags in text
    SUPPORTED_TAGS,          # List of 9 supported tags
    TAG_ALIASES,             # Dict of alias mappings
    AudioEffects,            # Speed/pitch effects
)
```

### generate() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Text to synthesize |
| `preset` | str | None | Preset name (natural, energetic, calm, narrator, character) |
| `audio_prompt_path` | str | None | Path to reference audio for voice cloning |
| `exaggeration` | float | 0.5 | Voice expressiveness (0.0-1.0) |
| `cfg_weight` | float | 0.5 | Classifier-free guidance weight |
| `temperature` | float | 0.8 | Sampling temperature |
| `speed` | float | 1.0 | Playback speed (0.5-2.0) |
| `pitch` | float | 0.0 | Pitch shift in semitones (-12 to +12) |

---

## License

Apache 2.0 (same as original Chatterbox)

---

## Acknowledgments

- [Resemble AI](https://www.resemble.ai/) for the original Chatterbox
- [NVIDIA](https://github.com/NVIDIA/BigVGAN) for BigVGAN v2
- [Silero](https://github.com/snakers4/silero-vad) for VAD model
