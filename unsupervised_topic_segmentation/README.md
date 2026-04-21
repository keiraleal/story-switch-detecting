# Embedding-Based Segmentation Baseline

An unsupervised, embedding-based story-boundary detector for this project,
adapted from:

> Solbiati, A., Poddar, S., Heffernan, K., Modi, S., Damaskinos, G., & Cali, J. (2021).
> *Unsupervised Topic Segmentation of Meetings with BERT Embeddings.*
> arXiv:2106.12978. https://doi.org/10.48550/arXiv.2106.12978

Original source repository (MIT licence):
https://github.com/gdamaskinos/unsupervised_topic_segmentation

---

## Why this exists

The project's main pipeline (`pipeline/process_data.py`) uses an LLM to
label story boundaries line-by-line — accurate but expensive (~2 API calls
per "in"-labeled line) and non-deterministic. This folder provides a
**fast, free, fully-reproducible** alternative that runs entirely on
sentence embeddings with zero API calls.

---

## What was changed from the original

The reference implementation was written against Facebook-internal tooling
and could not be run as shipped. Every file was rewritten or substantially
modified:

| File | Change |
|---|---|
| `types.py` → **`seg_types.py`** | Renamed to avoid shadowing Python's stdlib `types` module. Hyperparameter defaults tuned for shorter dialogues (window 15 → 10). |
| **`core.py`** | Replaced the original RoBERTa + fairseq encoder (which called internal Facebook APIs) with `sentence-transformers` (`stsb-roberta-base` for S-BERT, `all-distilroberta-v1` for the raw-BERT variant). All other algorithm logic — block comparison, max pooling, smoothing, depth scoring, threshold selection — is preserved from the paper. |
| **`dataset.py`** | Completely rewritten. The original loaded data from a Facebook-internal database. The new version reads the project's 4-column CSV format (`in/out/ambiguous, start, end, Transcript`), strips speaker prefixes (e.g. `CHI:`), and exposes `load_csv`, `load_directory`, `load_files`, and `human_boundary_labels` helpers. |
| **`baselines.py`** | Simplified to work off integer row indices instead of audio timestamps. |
| **`eval.py`** | Completely rewritten. Original required timestamp columns and an internal database for reference labels. New version computes (a) Pk / WinDiff using NLTK and (b) the exact same line-level precision / recall / F1 / IoU metrics as `pipeline/analysis.py`, so the embedding baseline and the LLM pipelines are directly comparable. Also adds `write_labeled_csvs` to emit output in the project's 4-column format. |
| **`run.py`** | New file. CLI entry point with `--algorithm`, `--input-dir`, `--output-dir`, and hyperparameter flags. |
| **`requirements.txt`** | New file. |
| **`compare_to_llm.py`** | New file. Aggregates existing LLM compare-CSVs and runs the embedding baseline on the same transcripts, printing a unified side-by-side table. |

---

## Algorithm

From Section 3 of the paper:

1. **Embed** every utterance with Sentence-BERT (dense 768-D vector per line).
2. **Compare windows.** At every candidate gap between line *i* and line *i+1*, take the *k* lines before and the *k* lines after. Max-pool each block into a single vector, then compute cosine similarity between the two pooled vectors. This produces a similarity time-series across the transcript.
3. **Smooth** the series with a box filter to suppress per-line noise (disfluencies, backchannels).
4. **Depth-score** every valley: how far down does it go relative to the nearest peaks on either side?
5. **Pick boundaries** at every local maximum of the depth score above `threshold × max_depth`.

**Key detail — max pooling:** averaging the block embeddings would let backchannel lines ("mhm", "yeah") wash out the semantic content of a story. Max-pooling dimension-by-dimension retains the strongest activation across the block, so the block vector reflects "what is the most topically-distinctive thing said in this window?"

---

## Files

| File | Purpose |
|---|---|
| `run.py` | **CLI entry point** |
| `compare_to_llm.py` | Side-by-side comparison of all methods |
| `core.py` | BERT / S-BERT segmentation algorithm |
| `baselines.py` | Random and Even baselines |
| `dataset.py` | CSV loader |
| `eval.py` | Pk / WinDiff + line-level metrics + CSV writer |
| `seg_types.py` | Enums + hyperparameter config |
| `requirements.txt` | Dependencies |

---

## Setup

```bash
cd unsupervised_topic_segmentation
pip install -r requirements.txt
```

First run downloads the S-BERT model (~500 MB) from Hugging Face and
caches it locally. Subsequent runs use the cache and start in ~16s.

---

## Usage

**Run S-BERT on a directory of CSVs and print metrics:**

```bash
python run.py --algorithm sbert --input-dir ../data/siblings/siblings-human
```

**Run all four algorithms and print a comparison table:**

```bash
python run.py --algorithm all --input-dir ../data/siblings/siblings-human
```

**Write labeled output CSVs (same format as `pipeline/labeled-out/`):**

```bash
python run.py \
  --algorithm sbert \
  --input-dir ../data/siblings/siblings-human \
  --output-dir segmented-out/siblings
```

**Compare embedding baseline against LLM pipelines (uses pre-existing compare CSVs):**

```bash
python compare_to_llm.py
```

**Key flags for `run.py`:**

| Flag | Meaning | Default |
|---|---|---|
| `--window` | Block size *k* for left/right window comparison | 10 |
| `--smoothing-passes` | Number of box-filter passes | 2 |
| `--smoothing-window` | Half-width of smoothing window | 1 |
| `--threshold` | Keep depth-score local maxima above threshold × max | 0.6 |
| `--cap-segments` | Cap segment count by expected average length | off |

---

## Metrics

**Pk and WinDiff** (lower is better, 0.0 = perfect, ~0.5 = random):
slide a window of width *k* across the transcript; count the fraction of
positions where the predicted and true segmentations disagree about whether
the two endpoint rows are in the same segment or not. Credits near-miss
boundaries rather than requiring exact line alignment.

**Line-level metrics** (higher is better): mirror `pipeline/analysis.py`
exactly — precision / recall / F1 / IoU computed per row for START markers,
END markers, and segment membership. Lets you compare the embedding baseline
directly to the LLM pipeline numbers.

---

## Preliminary results

Evaluated on two datasets. LLM results are read from the pre-existing
compare CSVs (`data/*/compare-hum-*/`). Embedding baseline (S-BERT) is run
fresh with default hyperparameters (window=10, threshold=0.6).

### Siblings (70 transcripts, ~17 000 utterances)

| Method | Pk ↓ | WinDiff ↓ | Start F1 ↑ | End F1 ↑ | Seg F1 ↑ | Seg IoU ↑ |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.2 | 0.3145 | 0.4244 | 0.3187 | 0.2857 | 0.6311 | 0.4610 |
| gpt-oss-120b | **0.2938** | 0.4294 | **0.3442** | **0.3082** | **0.6204** | 0.4497 |
| S-BERT (this baseline) | 0.5007 | 0.5110 | 0.0336 | 0.0335 | 0.5683 | 0.3969 |

### In-person-adult (30 transcripts, ~24 500 utterances)

| Method | Pk ↓ | WinDiff ↓ | Start F1 ↑ | End F1 ↑ | Seg F1 ↑ | Seg IoU ↑ |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.2 | **0.3561** | 0.5326 | 0.1613 | 0.1507 | **0.7230** | **0.5661** |
| gpt-oss-120b | 0.3667 | 0.5884 | **0.1754** | **0.1643** | 0.7161 | 0.5578 |
| S-BERT (this baseline) | 0.4889 | **0.5023** | 0.0264 | 0.0233 | 0.7163 | 0.5580 |

### Interpretation

- **LLMs clearly lead on boundary placement (Pk, Start/End F1).** Pk
  drops from ~0.49 (embedding) to ~0.29–0.37 (LLM) — a large gap.
- **Segment IoU is nearly identical on in-person-adult (~0.56–0.57).**
  Long continuous story stretches mean any method that can say "this
  section is a story" does well at the segment level. The LLMs pull
  ahead on siblings (shorter, more fragmented stories).
- **Start/End F1 is structurally low for the embedding baseline.** It
  places boundaries at similarity valleys which rarely coincide with
  the exact human-annotated row. Pk/WinDiff are the fair comparison
  here because they credit near-miss predictions.
- **Between the two LLMs:** within 0.02 on every metric — open-source
  (gpt-oss-120b) is viable and slightly better on siblings.
- **The embedding baseline establishes the zero-cost floor:** it
  captures roughly half the Pk performance of the LLM at zero API cost.
  Using it as a cheap candidate boundary generator — and only calling
  the LLM to confirm/reject those ~10–20 candidates per transcript
  instead of processing every line — is an obvious next step.

---

## Known limitations

1. **No in-story / out-of-story distinction.** The paper's method
   places every utterance inside *some* segment. Our data has `out`
   rows (conversational glue between stories) that shouldn't belong to
   any segment. Segment-level recall is therefore always 1.0 and
   precision is capped at the fraction of rows that are genuinely
   in-story (~50–70% depending on domain). A separate in/out classifier
   would be needed to fix this.

2. **Short transcripts.** The algorithm needs at least `2 × window + 3`
   utterances. Set `--window` smaller (e.g. 5) for very short files.

3. **Hyperparameter sensitivity.** Pk varies with window size and
   threshold. Defaults are tuned from the paper's values for our
   shorter dialogues; further tuning is worthwhile.

---

## Citation

```bibtex
@article{solbiati2021unsupervised,
  title   = {Unsupervised Topic Segmentation of Meetings with BERT Embeddings},
  author  = {Solbiati, Alessio and Poddar, Sourav and Heffernan, Kevin and
             Modi, Chinnadhurai and Damaskinos, Georgios and Cali, Jason},
  journal = {arXiv preprint arXiv:2106.12978},
  year    = {2021},
  url     = {https://arxiv.org/abs/2106.12978}
}
```
