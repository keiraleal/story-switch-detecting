# Story Switch Detecting

Automated detection of **story (narrative) boundaries** in conversational transcripts using LLMs, compared against human annotations to measure inter-rater agreement.

## Project Goal

Given a conversation transcript where a human annotator has marked which lines are "in" a story, "out" of a story, or "ambiguous," this project:

1. Sends the transcript through an LLM to independently predict where stories start and end.
2. Compares the LLM's predictions against the human ground truth.
3. Reports agreement metrics (accuracy, precision, recall, F1, IoU) to evaluate how well the LLM performs at this task.

The transcripts come from three conversational domains:
- **in-person-adult** — adult dyadic conversations (speakers labeled e.g. PHIL, BRAD)
- **siblings** — child–sibling play sessions (speakers labeled e.g. CHI, STE)
- **single-child** — child–adult interactions (speakers labeled e.g. NAN, CHI)

Two LLM models are evaluated against the same human labels:
- **GPT-5.2** (via OpenAI API)
- **GPT-OSS-120B** (via Hugging Face Inference)

## Repository Structure

```
Story Switch Detecting/
│
├── pipeline/                  ← MAIN PIPELINE (current version)
│   ├── process_data.py            Step 2: LLM labels story boundaries
│   ├── join.py                    Step 3: Merge over-segmented stories (original)
│   ├── join_fixed.py              Step 3: Merge over-segmented stories (fixed)
│   ├── analysis.py                Step 4: Compare human vs LLM labels
│   ├── validate_input.py          Step 1a: Check input file format
│   ├── fix_labels.py              Step 1b: Auto-fix common typos
│   ├── to-label/                  Input: human-labeled CSVs go here
│   ├── labeled-out/               Output from process_data.py
│   ├── joined-out/                Output from join.py / join_fixed.py
│   └── labeled-compare/           Output from analysis.py
│
├── data/                      ← COMPLETED DATASETS
│   ├── in-person-adult/
│   │   ├── in-person-adult-human/          Human annotations (ground truth)
│   │   ├── in-person-adult-gpt-5.2/        GPT-5.2 labels
│   │   ├── in-person-adult-gpt-oss-120b/   GPT-OSS-120B labels
│   │   ├── in-person-adult-compare-hum-5.2/    Human vs GPT-5.2 comparison
│   │   └── in-person-adult-compare-hum-oss/    Human vs GPT-OSS comparison
│   ├── siblings/
│   │   ├── siblings-human/
│   │   ├── siblings-gpt-5.2/
│   │   ├── siblings-gpt-oss-120b/
│   │   ├── siblings-compare-hum-5.2/
│   │   └── siblings-compare-hum-oss/
│   └── single-child/
│       ├── single-child-human/
│       ├── single-child-gpt-5.2/
│       ├── single-child-gpt-oss-120b/
│       ├── single-child-compare-hum-5.2/
│       └── single-child-compare-hum-oss/
│
├── v1/                        ← Version 1 (single-file, GPT-4o-mini)
├── v2/                        ← Version 2 (intermediate iteration)
├── test-and-dev/              ← Development / trial scripts
└── .env                       ← API keys (DO NOT COMMIT)
```

## Data Format

### Input / Human-Labeled CSVs

4 columns, CSV format:

| Column | Values | Description |
|--------|--------|-------------|
| `in/out/ambiguous` | `in`, `out`, `ambiguous` | Whether this line is part of a story |
| `start` | `TRUE`, `FALSE` | Whether this line is the START of a story segment |
| `end` | `TRUE`, `FALSE` | Whether this line is the END of a story segment |
| `Transcript` | text | The dialogue line (e.g. `PHIL: But anyways...`) |

Example:
```
in/out/ambiguous,start,end,Transcript
out,FALSE,FALSE,PHIL: But But anyways back to the the first thing.
out,FALSE,FALSE,BRAD: Okay.
in,TRUE,FALSE,BRAD: I've gotta pick up Pat.
in,FALSE,TRUE,BRAD: I dropped her off at the bookkeeper.
out,FALSE,FALSE,PHIL: .
```

### Comparison CSVs (output of analysis.py)

10 columns with side-by-side human and LLM labels:

| Column | Description |
|--------|-------------|
| `in/out/ambiguous` | Original human annotation |
| `start_human` | Human's start marker |
| `start_llm` | LLM's start marker |
| `end_human` | Human's end marker |
| `end_llm` | LLM's end marker |
| `Transcript` | The dialogue line |
| `story_seg_human` | TRUE if line falls inside any human-marked story |
| `story_seg_llm` | TRUE if line falls inside any LLM-marked story |
| `intersection` | TRUE if both human and LLM mark this line as in-story |
| `union` | TRUE if either human or LLM marks this line as in-story |

## Quick Start

### Prerequisites

- Python 3.8+
- A Hugging Face API token (for GPT-OSS-120B) or an OpenAI API key

### Setup

1. Create a `.env` file in the project root:
   ```
   HF_TOKEN=hf_your_token_here
   OPENAI_API_KEY=sk-your_key_here
   ```

2. Install dependencies:
   ```bash
   pip install openai python-dotenv
   ```

### Running the Pipeline

See `pipeline/README.md` for detailed step-by-step instructions.

```bash
cd pipeline

# 1. Place human-labeled CSVs in to-label/

# 2. (Optional) Validate and fix input files
python validate_input.py
python fix_labels.py

# 3. Run LLM labeling
python process_data.py

# 4. (Optional) Merge over-segmented stories
python join_fixed.py

# 5. Compare human vs LLM labels
python analysis.py
```

## Version History

| Version | Model | Notes |
|---------|-------|-------|
| v1 | GPT-4o-mini (OpenAI) | Single-file prototype, hardcoded paths |
| v2 | GPT-OSS-120B (HF) | Intermediate refactor with data mirroring |
| pipeline (current) | GPT-OSS-120B (HF) | Batch processing, validation, join step, full metrics |
