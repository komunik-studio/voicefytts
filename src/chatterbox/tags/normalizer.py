"""
Tag Normalization System for VoicefyTTS

Provides normalization and validation for Chatterbox's 9 native paralinguistic tags.
Supports 20+ aliases for common variations (e.g., [laughs] → [laugh]).
"""

import re
from typing import Tuple, List

# 9 native Chatterbox tags
SUPPORTED_TAGS = [
    "[laugh]",
    "[chuckle]",
    "[sigh]",
    "[gasp]",
    "[cough]",
    "[clear throat]",
    "[sniff]",
    "[groan]",
    "[shush]"
]

# Tag aliases for common variations
TAG_ALIASES = {
    # Laugh variations
    "[laughs]": "[laugh]",
    "[laughter]": "[laugh]",
    "[haha]": "[laugh]",
    "[hehe]": "[laugh]",
    
    # Chuckle variations
    "[chuckles]": "[chuckle]",
    
    # Sigh variations
    "[sighs]": "[sigh]",
    
    # Gasp variations
    "[gasps]": "[gasp]",
    
    # Cough variations
    "[coughs]": "[cough]",
    "[coughing]": "[cough]",
    
    # Throat clearing variations
    "[clears throat]": "[clear throat]",
    "[ahem]": "[clear throat]",
    
    # Sniff variations
    "[sniffs]": "[sniff]",
    "[sniffing]": "[sniff]",
    
    # Groan variations
    "[groans]": "[groan]",
    "[groaning]": "[groan]",
    "[ugh]": "[groan]",
    
    # Shush variations
    "[shh]": "[shush]",
    "[shhh]": "[shush]",
}


def normalize_tags(text: str) -> str:
    """
    Normalize tag aliases to their canonical forms.
    
    Args:
        text: Input text with potential tag aliases
        
    Returns:
        Text with all aliases replaced by canonical tags
        
    Example:
        >>> normalize_tags("Hello [laughs]!")
        "Hello [laugh]!"
        >>> normalize_tags("[ahem] Good morning")
        "[clear throat] Good morning"
    """
    result = text
    for alias, canonical in TAG_ALIASES.items():
        result = result.replace(alias, canonical)
    return result


def validate_tags(text: str) -> Tuple[bool, List[str]]:
    """
    Validate that all tags in text are supported.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, invalid_tags)
        - is_valid: True if all tags are supported
        - invalid_tags: List of unsupported tags found
        
    Example:
        >>> validate_tags("Hello [laugh] there")
        (True, [])
        >>> validate_tags("Hello [invalid] there")
        (False, ['[invalid]'])
    """
    # Find all tags in text (anything in square brackets)
    found_tags = re.findall(r'\[[^\]]+\]', text)
    
    # Check which tags are invalid (not in supported or aliases)
    invalid = [
        tag for tag in found_tags 
        if tag not in SUPPORTED_TAGS and tag not in TAG_ALIASES
    ]
    
    return len(invalid) == 0, invalid


def get_supported_tags() -> List[str]:
    """
    Get list of all supported canonical tags.
    
    Returns:
        List of supported tag strings
        
    Example:
        >>> tags = get_supported_tags()
        >>> "[laugh]" in tags
        True
    """
    return SUPPORTED_TAGS.copy()


def get_tag_aliases() -> dict:
    """
    Get dictionary of all tag aliases.
    
    Returns:
        Dictionary mapping aliases to canonical tags
        
    Example:
        >>> aliases = get_tag_aliases()
        >>> aliases["[laughs]"]
        "[laugh]"
    """
    return TAG_ALIASES.copy()


def __repr__() -> str:
    return f"TagNormalizer(supported={len(SUPPORTED_TAGS)}, aliases={len(TAG_ALIASES)})"
