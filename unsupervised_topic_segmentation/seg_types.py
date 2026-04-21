"""Enums and config types for the segmentation pipeline.

Renamed from the original ``types.py`` to avoid shadowing Python's
built-in ``types`` module.
"""

from enum import Enum
from typing import NamedTuple, Optional


class TopicSegmentationAlgorithm(Enum):
    """Which segmentation algorithm to run."""
    RANDOM = 0
    EVEN = 1
    BERT = 2      # Raw BERT/RoBERTa encoder
    SBERT = 3     # Sentence-BERT (recommended)


class TextTilingHyperparameters(NamedTuple):
    """Hyperparameters for the TextTiling-style boundary detection.

    Defaults are taken from Solbiati et al. (2021) but tuned slightly
    for the shorter dialogues in this project (meeting transcripts
    have hundreds to thousands of utterances; our transcripts are
    closer to 100-800 lines).
    """
    SENTENCE_COMPARISON_WINDOW: int = 10  # block size k for window-vs-window similarity
    SMOOTHING_PASSES: int = 2             # number of smoothing iterations
    SMOOTHING_WINDOW: int = 1             # half-width of smoothing window
    TOPIC_CHANGE_THRESHOLD: float = 0.6   # depth-score threshold (× max)


class TopicSegmentationConfig(NamedTuple):
    """Top-level segmentation config bundle."""
    TEXT_TILING: Optional[TextTilingHyperparameters] = None
    MAX_SEGMENTS_CAP: bool = False                   # set True to cap segment count
    MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH: int = 60
