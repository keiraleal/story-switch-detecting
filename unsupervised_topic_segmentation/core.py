"""Core segmentation algorithm from Solbiati et al. (2021).

Adapted from the original Facebook reference implementation for this
project's CSV format and Python environment.  Algorithm is unchanged;
what's changed is:

- Sentence-BERT (sentence_transformers) is used for embeddings instead
  of the original code's (broken) RoBERTa+fairseq mix.  This matches
  the paper's second tested embedding method (S-BERT) and is the one
  recommended for practical use in their implementation notes.
- Timestamp columns are replaced with row indices, since our CSV
  transcripts don't carry audio timestamps.
- Import of ``types`` renamed to ``seg_types`` to avoid shadowing the
  Python stdlib.

The algorithm (from Section 3.2 of the paper):

    1. Compute a sentence embedding for every utterance.
    2. Slide a window of size ``k`` along the embedding sequence.  For
       each candidate gap i between utterance i and i+1, form two
       block embeddings by max-pooling [i-k..i] and [i+1..i+k+1], then
       compute cosine similarity between them.  This produces a
       similarity time series.
    3. Smooth the series.
    4. Compute "depth scores" — how deep the local valley at each
       candidate gap is relative to its surrounding peaks.
    5. Pick local maxima of the depth score above a threshold; those
       are the topic-change indices.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import baselines as topic_segmentation_baselines
from seg_types import (
    TopicSegmentationAlgorithm,
    TopicSegmentationConfig,
)


# Global embedding models are loaded lazily on first use so that importing
# this module (e.g. from baselines.py or eval.py) stays cheap.
_SBERT_MODEL: SentenceTransformer | None = None
_BERT_MODEL: SentenceTransformer | None = None


def _get_sbert_model() -> SentenceTransformer:
    """Load Sentence-BERT (stsb-roberta-base, same as the paper)."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer("sentence-transformers/stsb-roberta-base")
    return _SBERT_MODEL


def _get_bert_model() -> SentenceTransformer:
    """Load a generic BERT-family encoder (roberta-base) via sentence-transformers.

    Sentence-transformers wraps a bare RoBERTa encoder with mean pooling
    over tokens, which matches the paper's averaging strategy for the
    non-S-BERT variant.
    """
    global _BERT_MODEL
    if _BERT_MODEL is None:
        _BERT_MODEL = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    return _BERT_MODEL


def depth_score(timeseries: List[float]) -> List[float]:
    """Compute depth scores for each interior point of a similarity time series.

    The depth score at position i is the sum of the gaps between i and
    the nearest peaks to its left and right.  Large depth scores mark
    valleys, which are candidate topic-boundary locations.
    """
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]:
            right += 1
        depth_scores.append(
            (timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i])
        )
    return depth_scores


def smooth(timeseries: List[float], n: int, s: int) -> List[float]:
    """Simple box-filter smoothing with ``n`` passes and half-width ``s``."""
    smoothed = list(timeseries)
    for _ in range(n):
        for index in range(len(smoothed)):
            neighbours = smoothed[max(0, index - s): min(len(timeseries) - 1, index + s)]
            if neighbours:
                smoothed[index] = sum(neighbours) / len(neighbours)
    return smoothed


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _block_embedding(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Max-pool over rows [start:end] of the embedding matrix.

    Max pooling (instead of mean) is the paper's key robustness trick:
    it lets semantically-rich tokens dominate over fillers/disfluencies.
    """
    block = embeddings[start:end]
    if block.shape[0] == 0:
        return np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
    return block.max(axis=0)


def block_comparison_score(embeddings: np.ndarray, k: int) -> List[float]:
    """Compute window-vs-window similarity at every candidate gap.

    For each gap index i in [k, N-k-1], compare a left block [i-k..i+1]
    against a right block [i+1..i+k+2] using max-pooled embeddings and
    cosine similarity.  Returns a 1-D list of similarities aligned to
    gap indices starting at i=k.
    """
    scores = []
    n = embeddings.shape[0]
    for i in range(k, n - k):
        left = _block_embedding(embeddings, i - k, i + 1)
        right = _block_embedding(embeddings, i + 1, i + k + 2)
        scores.append(_cosine(left, right))
    return scores


def get_local_maxima(array: List[float]):
    """Return (indices, values) of strict local maxima in a 1-D sequence."""
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values


def depth_score_to_topic_change_indexes(
    depth_score_timeseries: List[float],
    num_utterances: int,
    topic_segmentation_configs: TopicSegmentationConfig,
) -> List[int]:
    """Pick the topic-change gap indices from the depth-score series.

    Two selection modes are supported via ``MAX_SEGMENTS_CAP``:

    - When ``MAX_SEGMENTS_CAP`` is True: sort local maxima by depth,
      keep the top K where K is bounded by the number of utterances
      divided by an expected average segment length.  Useful for UI
      display.
    - When False (default): keep every local maximum whose depth exceeds
      ``TOPIC_CHANGE_THRESHOLD × max_depth``.  This matches the paper's
      Pk-optimised evaluation setup.
    """
    if not depth_score_timeseries:
        return []

    tt_cfg = topic_segmentation_configs.TEXT_TILING
    threshold = tt_cfg.TOPIC_CHANGE_THRESHOLD * max(depth_score_timeseries)

    local_maxima_indices, local_maxima = get_local_maxima(depth_score_timeseries)
    if not local_maxima:
        return []

    if topic_segmentation_configs.MAX_SEGMENTS_CAP:
        avg_seg = topic_segmentation_configs.MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH
        order = np.argsort(local_maxima)[::-1]
        sorted_maxima = [local_maxima[i] for i in order]
        sorted_indices = [local_maxima_indices[i] for i in order]

        cutoff = 0
        for cutoff in range(len(sorted_maxima)):
            if sorted_maxima[cutoff] <= threshold:
                break
        else:
            cutoff = len(sorted_maxima)

        max_segments = max(1, num_utterances // avg_seg)
        slice_length = min(max_segments, cutoff)
        kept = sorted_indices[:slice_length]
        return sorted(kept)

    return [
        idx for idx, m in zip(local_maxima_indices, local_maxima) if m > threshold
    ]


def _encode_utterances(
    sentences: List[str],
    algorithm: TopicSegmentationAlgorithm,
) -> np.ndarray:
    """Return an ``(N, D)`` numpy array of utterance embeddings."""
    if algorithm == TopicSegmentationAlgorithm.SBERT:
        model = _get_sbert_model()
    elif algorithm == TopicSegmentationAlgorithm.BERT:
        model = _get_bert_model()
    else:
        raise ValueError(f"Cannot encode with algorithm: {algorithm}")

    # Replace empty strings so the encoder doesn't emit NaNs.
    cleaned = [s if (isinstance(s, str) and s.strip()) else "." for s in sentences]
    with torch.no_grad():
        emb = model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
    return emb


def topic_segmentation_bert(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
    topic_segmentation_configs: TopicSegmentationConfig,
    algorithm: TopicSegmentationAlgorithm = TopicSegmentationAlgorithm.SBERT,
) -> Dict[str, List[int]]:
    """Run the BERT/S-BERT segmentation algorithm on every transcript in ``df``.

    Args:
        df: Concatenated transcripts.  Each row is one utterance.
        meeting_id_col_name: Column grouping rows into transcripts.
        start_col_name: Column containing the row's ordinal index (or
            timestamp).  Used only for reporting / duration.
        end_col_name: Same as ``start_col_name`` but for row end.
        caption_col_name: Column holding the utterance text.
        topic_segmentation_configs: Hyperparameters bundle.
        algorithm: BERT or SBERT.

    Returns:
        A dict mapping each meeting_id to the list of **gap indices**
        that were chosen as topic boundaries.  A gap index ``g`` means
        "there is a boundary between utterance g and utterance g+1"
        within that meeting.
    """
    tt = topic_segmentation_configs.TEXT_TILING

    segments: Dict[str, List[int]] = {}
    for meeting_id in sorted(set(df[meeting_id_col_name])):
        meeting_data = df[df[meeting_id_col_name] == meeting_id]
        sentences = list(meeting_data[caption_col_name].astype(str))
        n = len(sentences)

        if n < 2 * tt.SENTENCE_COMPARISON_WINDOW + 3:
            # Too short for this window size — skip boundary detection.
            segments[meeting_id] = []
            continue

        embeddings = _encode_utterances(sentences, algorithm)

        sim_series = block_comparison_score(embeddings, k=tt.SENTENCE_COMPARISON_WINDOW)
        sim_series = smooth(sim_series, n=tt.SMOOTHING_PASSES, s=tt.SMOOTHING_WINDOW)
        depth = depth_score(sim_series)

        local_change_positions = depth_score_to_topic_change_indexes(
            depth, n, topic_segmentation_configs
        )

        # ``depth`` is indexed over interior positions of ``sim_series``.
        # Convert (sim-series-index in depth) back to an absolute
        # utterance-gap index within the transcript:
        #   sim_series index i corresponds to gap (k + i)
        #   depth index j corresponds to sim_series index (j + 1)
        boundary_gaps = [
            tt.SENTENCE_COMPARISON_WINDOW + pos + 1 for pos in local_change_positions
        ]
        segments[meeting_id] = boundary_gaps

    return segments


def topic_segmentation(
    topic_segmentation_algorithm: TopicSegmentationAlgorithm,
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
    topic_segmentation_config: TopicSegmentationConfig,
) -> Dict[str, List[int]]:
    """Dispatch to the chosen segmentation algorithm."""
    if topic_segmentation_algorithm in (
        TopicSegmentationAlgorithm.BERT,
        TopicSegmentationAlgorithm.SBERT,
    ):
        return topic_segmentation_bert(
            df,
            meeting_id_col_name,
            start_col_name,
            end_col_name,
            caption_col_name,
            topic_segmentation_config,
            algorithm=topic_segmentation_algorithm,
        )
    if topic_segmentation_algorithm == TopicSegmentationAlgorithm.RANDOM:
        return topic_segmentation_baselines.topic_segmentation_random(
            df, meeting_id_col_name, start_col_name, end_col_name, caption_col_name
        )
    if topic_segmentation_algorithm == TopicSegmentationAlgorithm.EVEN:
        return topic_segmentation_baselines.topic_segmentation_even(
            df, meeting_id_col_name, start_col_name, end_col_name, caption_col_name
        )
    raise NotImplementedError(f"Algorithm not implemented: {topic_segmentation_algorithm}")
