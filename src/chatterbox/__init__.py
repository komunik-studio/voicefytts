try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

try:
    __version__ = version("voicefytts")
except Exception:
    __version__ = "0.1.6"  # Fallback version


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# VoicefyTTS specific exports
from .presets import get_preset, list_presets, GENERATION_PRESETS
from .tags import normalize_tags, validate_tags, SUPPORTED_TAGS, TAG_ALIASES
from .audio_effects import AudioEffects