"""Chatterbox Tag Normalization System"""

from .normalizer import (
    normalize_tags,
    validate_tags,
    get_supported_tags,
    get_tag_aliases,
    SUPPORTED_TAGS,
    TAG_ALIASES
)

__all__ = [
    "normalize_tags",
    "validate_tags",
    "get_supported_tags",
    "get_tag_aliases",
    "SUPPORTED_TAGS",
    "TAG_ALIASES"
]
