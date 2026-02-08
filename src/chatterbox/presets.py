"""
Generation Presets for VoicefyTTS

Provides 5 predefined configurations for common use cases:
- natural: General-purpose, balanced speech
- energetic: Upbeat content, advertisements
- calm: Meditation, relaxation
- narrator: Audiobooks, professional narration
- character: Animation, games, character voices
"""

from typing import Dict, Any, List


GENERATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "natural": {
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "speed": 1.0,
        "pitch": 0.0,
    },
    "energetic": {
        "exaggeration": 0.7,
        "cfg_weight": 0.4,
        "temperature": 0.9,
        "speed": 1.1,
        "pitch": 1.0,
    },
    "calm": {
        "exaggeration": 0.3,
        "cfg_weight": 0.6,
        "temperature": 0.7,
        "speed": 0.9,
        "pitch": -1.0,
    },
    "narrator": {
        "exaggeration": 0.4,
        "cfg_weight": 0.7,
        "temperature": 0.75,
        "speed": 0.95,
        "pitch": 0.0,
    },
    "character": {
        "exaggeration": 0.8,
        "cfg_weight": 0.3,
        "temperature": 1.0,
        "speed": 1.0,
        "pitch": 0.0,
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Get preset configuration by name.
    
    Args:
        name: Preset name (natural, energetic, calm, narrator, character)
    
    Returns:
        Dictionary with preset parameters
    
    Raises:
        ValueError: If preset name is not found
    
    Example:
        >>> preset = get_preset("energetic")
        >>> preset["speed"]
        1.1
    """
    if name not in GENERATION_PRESETS:
        available = ", ".join(GENERATION_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )
    return GENERATION_PRESETS[name].copy()


def list_presets() -> List[str]:
    """
    Return list of available preset names.
    
    Returns:
        List of preset names
    
    Example:
        >>> presets = list_presets()
        >>> "energetic" in presets
        True
    """
    return list(GENERATION_PRESETS.keys())


def get_preset_description(name: str) -> str:
    """
    Get human-readable description of a preset.
    
    Args:
        name: Preset name
    
    Returns:
        Description string
    
    Example:
        >>> desc = get_preset_description("calm")
        >>> "Meditation" in desc
        True
    """
    descriptions = {
        "natural": "General-purpose, balanced speech for everyday use",
        "energetic": "Upbeat and dynamic for advertisements and motivational content",
        "calm": "Soothing and relaxed for meditation and ASMR",
        "narrator": "Professional and clear for audiobooks and documentaries",
        "character": "Highly expressive for animation and game characters",
    }
    
    if name not in descriptions:
        raise ValueError(f"Unknown preset '{name}'")
    
    return descriptions[name]
