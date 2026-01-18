import csv
import os
import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI API
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use Hugging Face Inference Providers for GPT-OSS
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories (relative to script location)
INPUT_DIR = os.path.join(SCRIPT_DIR, "to-label")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "labeled-out")


def llm_summary(content):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # For HF Inference use: "openai/gpt-oss-120b"
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
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # For HF Inference use: "openai/gpt-oss-120b"
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
    """Create a copy of input file with start and end columns set to FALSE."""
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
    """Process a single transcript file."""
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
