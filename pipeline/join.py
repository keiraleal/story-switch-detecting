"""
Segment Merging / Join Step (Step 3 of the pipeline — original version)

Reads LLM-labeled transcripts from labeled-out/ and merges adjacent story
segments that the LLM incorrectly split apart (over-segmentation fix).

Algorithm:
    1. Scan the labeled output for start/end markers identifying segments.
    2. For the first segment found, summarize it with the LLM.
    3. For each subsequent segment, summarize it and ask the LLM whether
       the two summaries describe the same story (llm_combine_stories).
       - If TRUE (same story): erase the boundary between them (remove
         the end marker of segment 1 and the start marker of segment 2).
       - If FALSE (different stories): move on to compare the next pair.
    4. Write the result to joined-out/*_joined.csv.

NOTE: This version uses a for-loop and reassigns `i` inside the loop body
      to attempt backtracking after a merge.  Because Python for-loops
      ignore manual changes to the loop variable, merged segments may not
      be re-checked correctly.  See join_fixed.py for the corrected version
      that uses a while-loop.

Input:  labeled-out/*_labeled.csv   (from process_data.py)
Output: joined-out/*_joined.csv

Dependencies:
    - openai (via Hugging Face Inference)
    - python-dotenv

Usage:
    python join.py
"""

import os
import csv
import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# To use OpenAI directly, uncomment and swap with the HF block below.
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "labeled-out")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "joined-out")

def llm_summary(content):
    """Summarize a transcript excerpt (same prompt as process_data.py)."""
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

def llm_combine_stories(summary1, summary2):
    """Ask the LLM whether two segment summaries describe the same story.

    Args:
        summary1: Summary of the first (earlier) segment.
        summary2: Summary of the second (later) segment.

    Returns:
        'TRUE' if the LLM considers them the same story, 'FALSE' otherwise.
    """
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": f'''Consider the following summary of a story: {summary1}. \n Now
            consider the following summary of a story: {summary2}. \n Your job is to consider whether
            or not the provided summaries are part of the same story. If they are, output 'TRUE'. If
            they are not, output 'FALSE'. Do not output anything else. The summarized stories can be
            part of the same story if they are continuation of each other. The summaries were provided
            in the order in which they appear. Stories contain a consistent train of thought.'''}
        ]
    )
    return response.choices[0].message.content

def create_output_file(input_path, output_path):
    """Copy the labeled CSV to the output path (preserving existing labels)."""
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
            writer.writerow(row)
            index += 1


def update_output_row(output_path, row_index, start_value=None, end_value=None):
    """Update a specific row in the output file."""
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
    # Create the output file
    create_output_file(input_path, output_path)
    
    # Read the rows
    with open(input_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    summary1 = 'EMP'
    start_index = None
    end_index = None
    
    for i in range(1, len(rows)):
        if rows[i][1] == 'TRUE':
            start_index = i
            print(f"  Line {i}: Found segment START")
        if rows[i][2] == 'TRUE':
                print(f"  Line {i}: Found segment END (segment: lines {start_index}-{i})")
                if summary1 == 'EMP':
                    print(f"    Summarizing first segment...")
                    summary1 = llm_summary('\n'.join(row[3] for row in rows[start_index:i+1]))
                    end_index = i
                    print(f"    Summary1: {summary1[:80]}...")
                else:
                    print(f"    Summarizing second segment...")
                    summary2 = llm_summary('\n'.join(row[3] for row in rows[start_index:i+1]))
                    print(f"    Summary2: {summary2[:80]}...")
                    print(f"    Comparing summaries...")
                    combine_stories = llm_combine_stories(summary1, summary2)
                    print(f"    Same story? {combine_stories}")
                    if combine_stories == 'TRUE':
                        print(f"    MERGING: Removing boundary at lines {end_index} (end) and {start_index} (start)")
                        update_output_row(output_path, start_index, start_value='FALSE')
                        update_output_row(output_path, end_index, end_value='FALSE')
                        summary1 = 'EMP'
                        i = start_index
                        print(f"    Resetting to line {start_index} to re-scan merged segment")
                    else:
                        summary1 = summary2
                        print(f"    Different stories. Moving on.")
                        


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    csv_files.sort()
    
    # Process each CSV file
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        output_path = os.path.join(OUTPUT_DIR, filename.replace('_labeled', '_joined'))
        print(f"Processing: {filename}")
        process_transcript(csv_file, output_path)

if __name__ == "__main__":
    main()