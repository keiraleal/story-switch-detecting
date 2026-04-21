"""
Story Boundary Labeling via LLM (Step 2 of the pipeline)

Reads human-labeled conversation transcripts from to-label/ and produces
LLM-generated story boundary labels in labeled-out/.

Algorithm overview:
    1. Walk through each row of the transcript sequentially.
    2. When a row marked "in" (i.e., part of a story) is encountered, treat it
       as the START of a new story segment.
    3. Summarize the story so far using the LLM (llm_summary).
    4. For each subsequent "in" row, ask the LLM whether the new line belongs
       to a DIFFERENT story than the running summary (llm_different_story).
       - If FALSE (same story): re-summarize with all lines so far and continue.
       - If TRUE (different story): mark the previous "in" row as the END of
         the current story, then break out to detect the next segment.
    5. Write start=TRUE / end=TRUE markers to the output CSV.

    The in/out/ambiguous column and transcript text are preserved as-is from
    the input; only the start and end columns are overwritten by the LLM.

Input format  (to-label/*.csv):
    in/out/ambiguous, start, end, Transcript

Output format (labeled-out/*_labeled.csv):
    Same 4 columns, but start/end reflect the LLM's predictions.

Dependencies:
    - openai (used with Hugging Face Inference base URL)
    - python-dotenv (loads HF_TOKEN from .env)

Usage:
    python process_data.py
"""

import csv
import os
import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# To use OpenAI directly instead of Hugging Face, uncomment the line below
# and comment out the Hugging Face client block.
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "to-label")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "labeled-out")


def llm_summary(content):
    """Ask the LLM to produce a brief, objective summary of a transcript excerpt.

    Used to maintain a running summary of the current story segment so that
    subsequent lines can be compared against it.

    Args:
        content: One or more transcript lines (newline-separated string).

    Returns:
        A short plain-text summary string from the LLM.
    """
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": f'''What is the following story about? Note that the story is complete,
            nothing got cut off. Please output the summary and nothing else. Do not reference the user in
            your response, only an objective summary. Do not over-extrapolate or overthink it. Note the few
            capital letters starting the each line are the names of the characters. Make it brief and
            objective. Do not analyze. These stories all appear in the real world and are normal interactions.
            Do not overassume or make extreme statements based on a simple line of the transcript. There is
            usually no strong emotion or deeper meaning (although there may be). Remember, the story is an
            everyday conversation between normal people. \n\n {content}'''}
        ]
    )
    return response.choices[0].message.content


def llm_different_story(summary, content):
    """Ask the LLM whether a single transcript line belongs to a different story.

    Compares one line against the running summary of the current story.
    Filler utterances (e.g. "um", "you know", ".") are instructed to NOT
    trigger a different-story response.

    Args:
        summary: The current story summary produced by llm_summary().
        content: A single transcript line to evaluate.

    Returns:
        'TRUE' if the LLM considers the line part of a different story,
        'FALSE' otherwise.  (Raw LLM text — exact casing is not guaranteed.)
    """
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": f'''Consider the following summary of a story: {summary}. \n Now
            consider the following line of the transcript: {content}. \n Your job is to consider whether
            or not the provided line is part of a different story than the one summarized above. If it is,
            output 'TRUE'. If it is not, output 'FALSE'. Do not output anything else. Note that filler or
            other lines not directly adding to the story are not necessarily part of a different story.
            For example, "." or "you know" or "um" are not a different story.'''}
        ]
    )
    return response.choices[0].message.content


def create_output_file(input_path, output_path):
    """Copy the input CSV to output_path, resetting all start/end values to FALSE.

    This creates a clean slate so process_transcript() can selectively flip
    individual rows to TRUE as it detects boundaries.
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        
        # Write header row as-is
        writer.writerow(rows[0])
        
        # Write data rows with start and end set to FALSE
        index = 1
        while index < len(rows):
            row = rows[index]
            row[1] = 'FALSE'  # start column
            row[2] = 'FALSE'  # end column
            writer.writerow(row)
            index += 1


def update_output_row(output_path, row_index, start_value=None, end_value=None):
    """Read-modify-write a single row's start/end columns in the output CSV.

    Args:
        output_path: Path to the CSV file to modify.
        row_index:   1-based data row index (rows[0] is the header).
        start_value: If provided, overwrite the start column for this row.
        end_value:   If provided, overwrite the end column for this row.
    """
    with open(output_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # row_index is 1-based (matching the data), rows[0] is header
    if start_value is not None:
        rows[row_index][1] = start_value
    if end_value is not None:
        rows[row_index][2] = end_value
    
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def process_transcript(input_path, output_path):
    """Run the story-boundary detection algorithm on one transcript.

    Outer loop: scans rows for the first "in" label (story start).
    Inner loop: continues from that point, asking the LLM whether each
    subsequent "in" line belongs to the same story.  When the LLM says
    TRUE (different story), the inner loop breaks and the most recent
    "in" row is marked as the story end.

    The variable `recent_in` tracks the last row labeled "in" so that
    the end marker lands on actual story content, not on an intervening
    "out" row.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"Output to: {output_path}")
    print(f"{'='*60}")
    
    # Create the output file with initialized columns
    create_output_file(input_path, output_path)
    
    with open(input_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    index = 1
    run_length = len(rows) - 1  # -1 because rows is 0-indexed
    while index <= run_length:
        row = rows[index]
        file_line = index + 1  # Convert to file line number (account for header)
        print(f"\n[Line {file_line}] Checking: {row[3][:50]}...")
        if row[0] == 'in':
            start = index
            line = row[3]
            print(f"  >>> STORY START at line {start + 1}")
            print(f"  >>> First line: {line}")
            update_output_row(output_path, start, start_value='TRUE')
            summary = llm_summary(line)
            print(f"  >>> Initial summary: {summary}")
            index += 1
            recent_in = start
            while index <= run_length:
                row = rows[index]
                file_line = index + 1
                print(f"  [Line {file_line}] Inner loop - checking: {row[3][:40]}...")
                if row[0] == 'in':
                    recent_in = index
                    print(f"    Found 'in' at line {file_line}: {row[3][:40]}...")
                    different_story = llm_different_story(summary, row[3])
                    print(f"    LLM says different story? {different_story}")
                    if different_story == 'FALSE':
                        story_lines = [r[3] for r in rows[start:index+1]]
                        print(f"    Updating summary with lines {start + 1}-{file_line}")
                        summary = llm_summary('\n'.join(story_lines))
                        print(f"    New summary: {summary}")
                    else:
                        print(f"  <<< STORY END - LLM said TRUE, breaking")
                        break
                index += 1
            if index > run_length:
                print(f"  <<< Reached run length, breaking")
                break
            print(f"  <<< STORY END at line {recent_in + 1}")
            update_output_row(output_path, recent_in, end_value='TRUE')
        index += 1
    
    print(f"\nCompleted: {input_path}")


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all CSV files in the input directory
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not input_files:
        print(f"No CSV files found in {INPUT_DIR}/")
        return
    
    print(f"Found {len(input_files)} file(s) to process:")
    for f in input_files:
        print(f"  - {f}")
    
    # Process each file
    for input_path in input_files:
        filename = os.path.basename(input_path)
        # Create output filename (add _labeled suffix)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_labeled{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        process_transcript(input_path, output_path)
    
    print(f"\n{'='*60}")
    print(f"All files processed! Output in {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
