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
| **Vocoder** | HiFi-GAN | BigVGAN v2 (NVIDIA) - *Coming in Phase 2* |
| **Speed Control** | ❌ Not available | ✅ 0.5x - 2.0x via post-processing |
| **Pitch Control** | ❌ Not available | ✅ -12 to +12 semitones |
| **Presets** | Manual tuning only | ✅ 5 presets (natural, energetic, calm, narrator, character) |
| **Fine-tuning** | Separate toolkit | ✅ Integrated training scripts |

### 🔧 Phase 1 Changes (Current)

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

# Load model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with tags
text = "Hi there [chuckle], how are you doing today?"
wav = model.generate(text, audio_prompt_path="reference.wav")

ta.save("output.wav", wav, model.sr)
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

VoicefyTTS supports the same 9 paralinguistic tags as Chatterbox:

| Tag | Description | Example |
|-----|-------------|---------|
| `[laugh]` | Laughter | "That's hilarious [laugh]!" |
| `[chuckle]` | Light laugh | "Well [chuckle], that's funny" |
| `[sigh]` | Sigh | "[sigh] What a long day..." |
| `[gasp]` | Surprise | "[gasp] I can't believe it!" |
| `[cough]` | Cough | "[cough] Excuse me" |
| `[clear throat]` | Throat clearing | "[clear throat] Good morning" |
| `[sniff]` | Sniffing | "[sniff] Thank you..." |
| `[groan]` | Groan | "[groan] This is difficult" |
| `[shush]` | Shushing | "[shush] Be quiet!" |

---

## Roadmap

- [x] **Phase 1**: Remove Perth watermark, add Voicefy placeholder
- [ ] **Phase 2**: Integrate BigVGAN v2 vocoder
- [ ] **Phase 3**: Tag normalization system
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
