# VoicefyTTS

> **Fork of [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI**
> 
> Enhanced with BigVGAN v2 vocoder, speed/pitch controls, and Voicefy watermarking.

[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

---

## 🎯 What's Different from Original Chatterbox?

VoicefyTTS is a production-ready fork of Chatterbox with the following enhancements:

### ✅ Changes in This Fork

| Feature | Original Chatterbox | VoicefyTTS |
|---------|---------------------|------------|
| **Watermark** | Perth (Resemble AI) | Voicefy (placeholder, disabled by default) |
| **Vocoder** | HiFi-GAN | ✅ **BigVGAN v2 (NVIDIA)** - 4.36+ PESQ, ~5760 kHz |
| **Speed Control** | ❌ Not available | ⏳ 0.5x - 2.0x via post-processing *(Phase 4)* |
| **Pitch Control** | ❌ Not available | ⏳ -12 to +12 semitones *(Phase 4)* |
| **Presets** | Manual tuning only | ⏳ 5 presets (natural, energetic, calm, narrator, character) *(Phase 5)* |
| **Fine-tuning** | Separate toolkit | ⏳ Integrated training scripts *(Phase 6)* |

### 🔧 Recent Changes

**Phase 2 (✅ Complete):**
- **BigVGAN v2 Integration**: NVIDIA's high-quality vocoder (PESQ 4.36+ vs 4.0, ~5760 kHz vs ~1000 kHz)
- **Vocoder Selection**: Choose between BigVGAN (quality) or HiFi-GAN (fallback)
- **Auto-fallback**: Gracefully falls back to HiFi-GAN if BigVGAN unavailable

**Phase 1 (✅ Complete):**
- **Removed Perth Watermark**: Eliminated `resemble-perth` dependency
- **Added Voicefy Watermark Placeholder**: Disabled by default, ready for future implementation
- **Clean Audio Output**: No watermarking applied unless explicitly enabled

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

---

## Usage

### Basic TTS (Same as Chatterbox)

```python
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load model (uses BigVGAN v2 by default)
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with tags
text = "Hi there [chuckle], how are you doing today?"
wav = model.generate(text, audio_prompt_path="reference.wav")

ta.save("output.wav", wav, model.sr)
```

### 🆕 Vocoder Selection (Phase 2)

```python
from chatterbox.models.s3gen import S3Token2Wav

# Use BigVGAN v2 (default - best quality)
s3gen = S3Token2Wav(vocoder_type='bigvgan', bigvgan_preset='quality')

# Use BigVGAN fast preset (faster inference)
s3gen = S3Token2Wav(vocoder_type='bigvgan', bigvgan_preset='fast')

# Use HiFi-GAN (fallback)
s3gen = S3Token2Wav(vocoder_type='hifigan')
```

### 🆕 Speed & Pitch Controls (Phase 4 - Coming Soon)

```python
# Generate with speed and pitch adjustments
wav = model.generate(
    text="Hello, this is a test",
    audio_prompt_path="reference.wav",
    speed=1.2,              # 20% faster
    pitch_semitones=2       # 2 semitones higher
)
```

### 🆕 Presets (Phase 5 - Coming Soon)

```python
from chatterbox.presets import get_preset

# Use preset configurations
preset = get_preset("energetic")
wav = model.generate(text, **preset)
```

---

## Supported Tags

VoicefyTTS supports the same 9 paralinguistic tags as Chatterbox, **plus 20+ aliases**:

| Tag | Description | Aliases | Example |
|-----|-------------|---------|---------|
| `[laugh]` | Laughter | `[laughs]`, `[laughter]`, `[haha]`, `[hehe]` | "That's hilarious [laugh]!" |
| `[chuckle]` | Light laugh | `[chuckles]` | "Well [chuckle], that's funny" |
| `[sigh]` | Sigh | `[sighs]` | "[sigh] What a long day..." |
| `[gasp]` | Surprise | `[gasps]` | "[gasp] I can't believe it!" |
| `[cough]` | Cough | `[coughs]`, `[coughing]` | "[cough] Excuse me" |
| `[clear throat]` | Throat clearing | `[clears throat]`, `[ahem]` | "[clear throat] Good morning" |
| `[sniff]` | Sniffing | `[sniffs]`, `[sniffing]` | "[sniff] Thank you..." |
| `[groan]` | Groan | `[groans]`, `[groaning]`, `[ugh]` | "[groan] This is difficult" |
| `[shush]` | Shushing | `[shh]`, `[shhh]` | "[shush] Be quiet!" |

**Tag Normalization (Phase 3):**
- Aliases are automatically converted to canonical forms
- Example: `"Hi [laughs]!"` → `"Hi [laugh]!"`
- Works with all TTS models (standard, turbo, multilingual)

---

## Roadmap

- [x] **Phase 1**: Remove Perth watermark, add Voicefy placeholder
- [x] **Phase 2**: Integrate BigVGAN v2 vocoder
- [x] **Phase 3**: Tag normalization system
- [ ] **Phase 4**: Speed/Pitch post-processing
- [ ] **Phase 5**: Generation presets
- [ ] **Phase 6**: Fine-tuning toolkit
- [ ] **Phase 7**: Backend integration (tts-worker)
- [ ] **Phase 8**: Frontend controls
- [ ] **Phase 9**: Documentation
- [ ] **Phase 10**: Full verification

---

## Credits

**Original Chatterbox** by [Resemble AI](https://resemble.ai)

**VoicefyTTS Fork** maintained by [Komunik Studio](https://github.com/komunik-studio)

### Acknowledgements
- [Chatterbox](https://github.com/resemble-ai/chatterbox) - Original TTS model
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) - NVIDIA vocoder
- [Podonos](https://podonos.com) - Speech evaluation platform

---

## License

Same as original Chatterbox (see LICENSE file)

## Disclaimer

This is a fork for production use. Don't use this model for harmful purposes.
