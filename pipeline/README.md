# Pipeline — Story Boundary Detection

This is the main (current) version of the story-switch detection pipeline.
It processes batches of human-labeled conversation transcripts through an LLM,
then compares the LLM's story boundary predictions against the human annotations.

## Pipeline Steps

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1 (optional): Validate & fix input files              │
│    validate_input.py  →  reports format errors              │
│    fix_labels.py      →  auto-corrects typos in-place       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                     to-label/*.csv
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 2: LLM labeling                                       │
│    process_data.py                                          │
│    Reads each transcript line-by-line.  When it encounters  │
│    an "in" line, it treats that as a story start, then asks │
│    the LLM about each subsequent "in" line: "is this part  │
│    of the same story?"  When the LLM says it's a different  │
│    story, the previous "in" line is marked as the end.      │
│                                                             │
│    API calls per transcript: ~2 per "in" line (1 summary +  │
│    1 comparison), so a 700-line transcript with 200 "in"    │
│    lines ≈ 400 API calls.                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    labeled-out/*_labeled.csv
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 3 (optional): Merge over-segmented stories            │
│    join_fixed.py  (preferred — uses while-loop)             │
│    join.py        (original — has a for-loop bug)           │
│                                                             │
│    Iterates over consecutive segments.  Summarizes each     │
│    pair and asks the LLM if they're the same story.         │
│    If yes, erases the boundary between them.                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    joined-out/*_joined.csv
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 4: Analysis / comparison                              │
│    analysis.py                                              │
│    Reads human labels (to-label/) and LLM labels            │
│    (labeled-out/ or joined-out/).  Produces:                │
│    - A comparison CSV per transcript                        │
│    - Per-file and overall metrics printed to stdout          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    labeled-compare/*_compare.csv
```

## Directory Layout

```
pipeline/
├── process_data.py        LLM story boundary labeling
├── join.py                Segment merging (original, has bug)
├── join_fixed.py          Segment merging (corrected)
├── analysis.py            Human vs LLM comparison & metrics
├── validate_input.py      Input format checker
├── fix_labels.py          Auto-fix typos in input files
│
├── to-label/              INPUT — place human-labeled CSVs here
├── labeled-out/           OUTPUT of process_data.py
├── joined-out/            OUTPUT of join.py / join_fixed.py
└── labeled-compare/       OUTPUT of analysis.py
```

## Step-by-Step Usage

### 1. Prepare input files

Place your human-labeled CSV files into `to-label/`. Each file must have
4 columns:

```
in/out/ambiguous,start,end,Transcript
out,FALSE,FALSE,PHIL: But But anyways back to the the first thing.
in,TRUE,FALSE,BRAD: I've gotta pick up Pat.
in,FALSE,TRUE,BRAD: I dropped her off at the bookkeeper.
```

**Column definitions:**

| Column | Valid values | Meaning |
|--------|-------------|---------|
| `in/out/ambiguous` | `in`, `out`, `ambiguous` | Whether this line is part of a story |
| `start` | `TRUE`, `FALSE` | First line of a story segment |
| `end` | `TRUE`, `FALSE` | Last line of a story segment |
| `Transcript` | any text | Dialogue line, prefixed with speaker name in caps |

### 2. (Optional) Validate input files

```bash
python validate_input.py
```

This checks every file in `to-label/` for:
- Correct number of columns (4)
- Valid header names
- Valid values in each column
- Non-empty transcript text

If errors are found, you can try auto-fixing them:

```bash
python fix_labels.py
```

This corrects common issues:
- Misspelled headers ("ambigious", "Trancript", etc.)
- Typos in the in/out column ("oin" → "in", "pout" → "out")
- Corrupted boolean values ("RUE" → "TRUE", "" → "FALSE")
- Alternate column orderings (reorders to the expected layout)
- Extra trailing columns (strips them)

Run `validate_input.py` again afterward to confirm all issues are fixed.

### 3. Run LLM labeling

```bash
python process_data.py
```

**What it does:**
- Reads each CSV from `to-label/`
- Creates a copy in `labeled-out/` with all start/end columns reset to FALSE
- Walks through the transcript line-by-line:
  - When it finds an `in` row → marks it as a story START
  - Summarizes the story so far using the LLM
  - For each subsequent `in` row, asks the LLM: "Is this line part of a different story?"
  - When the LLM says TRUE → marks the previous `in` row as the story END
  - Continues scanning for the next story
- Output files are named `{original_name}_labeled.csv`

**Requirements:**
- `.env` file with `HF_TOKEN` (for Hugging Face) or `OPENAI_API_KEY`
- To switch models, edit the `client` initialization and `model` parameter
  at the top of the file

**Note:** This step makes many API calls (roughly 2 per "in"-labeled line)
and can take several minutes per transcript.

### 4. (Optional) Merge over-segmented stories

```bash
python join_fixed.py
```

Use `join_fixed.py`, not `join.py` — the original has a loop bug where
reassigning the iterator variable inside a for-loop has no effect in Python.

**What it does:**
- Reads each file from `labeled-out/`
- For consecutive story segments, summarizes each and asks the LLM whether
  they describe the same story
- If yes: removes the boundary between them (erases the end marker of
  segment A and the start marker of segment B)
- Output files are named `{original_name}_joined.csv` in `joined-out/`

### 5. Run analysis

```bash
python analysis.py
```

**What it does:**
- For each file in `to-label/`, finds the corresponding LLM output
  (tries `_labeled` first, then `_joined`, then exact name match)
- Merges them into a 10-column comparison CSV in `labeled-compare/`
- Computes three families of metrics:

**Start detection** — exact line-level match of start=TRUE markers:
- TP: Both human and LLM mark start=TRUE on the same line
- FP: LLM marks start=TRUE where human did not
- FN: Human marks start=TRUE where LLM did not

**End detection** — same logic for end=TRUE markers.

**Segment overlap** — whether each line falls inside any story segment:
- A line is "in-segment" from the row where start=TRUE through the row
  where end=TRUE (both endpoints inclusive)
- IoU = intersection / union of in-segment lines

All metrics are reported per-file and aggregated across all files.

**To compare against joined output instead of raw labeled output:**
Change `LLM_LABELED_DIR` at the top of `analysis.py` from `labeled-out`
to `joined-out`.

## Metrics Glossary

| Metric | Formula | What it measures |
|--------|---------|------------------|
| Accuracy | (TP + TN) / total | Overall correctness |
| Precision | TP / (TP + FP) | Of all LLM-positive predictions, how many were correct |
| Recall | TP / (TP + FN) | Of all actual positives, how many the LLM found |
| F1 | 2 × (Prec × Rec) / (Prec + Rec) | Harmonic mean of precision and recall |
| IoU | intersection / union | Overlap between human and LLM story regions |

## Switching LLM Models

The pipeline currently uses **GPT-OSS-120B** via Hugging Face Inference.
To change the model:

1. Edit the `client` initialization in `process_data.py` (and `join.py` /
   `join_fixed.py` if using the join step):

   ```python
   # For OpenAI directly:
   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   # Then set model="gpt-5.2" (or another model) in the API calls

   # For Hugging Face Inference:
   client = OpenAI(
       base_url="https://router.huggingface.co/v1",
       api_key=os.getenv("HF_TOKEN")
   )
   # Then set model="openai/gpt-oss-120b" in the API calls
   ```

2. Update the `model=` parameter in `llm_summary()`, `llm_different_story()`,
   and `llm_combine_stories()`.

## Copying Data Into the Pipeline

The `data/` directory at the project root contains completed datasets
organized by domain and model. To run the pipeline on a dataset:

```bash
# Example: re-run on the in-person-adult data
cp data/in-person-adult/in-person-adult-human/*.csv pipeline/to-label/
python pipeline/process_data.py
python pipeline/analysis.py
```

## Known Limitations

- **API cost**: The line-by-line approach makes ~2 LLM calls per "in" row.
  A 700-line transcript with 200 "in" lines costs ~400 API calls.
- **Non-determinism**: LLM outputs vary between runs, so results are not
  perfectly reproducible.
- **Exact string matching**: The code checks `== 'TRUE'` exactly. If the
  LLM returns "True", "true", or "TRUE." it won't match. The join step
  is particularly sensitive to this.
- **join.py bug**: The original `join.py` uses a for-loop and tries to
  reassign the loop variable, which Python ignores. Use `join_fixed.py`.
