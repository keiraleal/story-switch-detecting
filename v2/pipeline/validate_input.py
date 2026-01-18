import csv
import os
import glob

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory (relative to script location)
INPUT_DIR = os.path.join(SCRIPT_DIR, "to-label")

# Expected format
EXPECTED_COLUMNS = ['in/out/ambiguous', 'start', 'end', 'Transcript']
VALID_IN_OUT_VALUES = ['in', 'out', 'ambiguous']
VALID_BOOL_VALUES = ['TRUE', 'FALSE']


def validate_file(filepath):
    """Validate a single CSV file format."""
    filename = os.path.basename(filepath)
    errors = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
    except Exception as e:
        print(f"  ❌ {filename}: Failed to read file - {e}")
        return False, 0
    
    # Check if file is empty
    if len(rows) == 0:
        print(f"  ❌ {filename}: File is empty")
        return False, 0
    
    # Check header
    header = rows[0]
    if len(header) != 4:
        errors.append(f"Expected 4 columns, found {len(header)}")
    elif header != EXPECTED_COLUMNS:
        errors.append(f"Header mismatch. Expected {EXPECTED_COLUMNS}, got {header}")
    
    # Check data rows
    for i, row in enumerate(rows[1:], start=2):  # start=2 for file line numbers
        # Check column count
        if len(row) != 4:
            errors.append(f"Line {i}: Expected 4 columns, found {len(row)}")
            continue
        
        # Check in/out/ambiguous column
        if row[0] not in VALID_IN_OUT_VALUES:
            errors.append(f"Line {i}: Invalid value '{row[0]}' in column 1 (expected: {VALID_IN_OUT_VALUES})")
        
        # Check start column
        if row[1] not in VALID_BOOL_VALUES:
            errors.append(f"Line {i}: Invalid value '{row[1]}' in start column (expected: TRUE/FALSE)")
        
        # Check end column
        if row[2] not in VALID_BOOL_VALUES:
            errors.append(f"Line {i}: Invalid value '{row[2]}' in end column (expected: TRUE/FALSE)")
        
        # Check transcript is not empty
        if len(row[3].strip()) == 0:
            errors.append(f"Line {i}: Transcript column is empty")
    
    # Report results
    row_count = len(rows) - 1  # Exclude header
    if errors:
        print(f"  ❌ {filename}: {len(errors)} error(s) found ({row_count} data rows)")
        for error in errors[:10]:  # Show first 10 errors
            print(f"      - {error}")
        if len(errors) > 10:
            print(f"      ... and {len(errors) - 10} more errors")
        return False, row_count
    else:
        print(f"  ✓ {filename}: Valid ({row_count} data rows)")
        return True, row_count


def main():
    print(f"{'='*60}")
    print("INPUT FILE VALIDATION")
    print(f"{'='*60}")
    print(f"Checking files in: {INPUT_DIR}/\n")
    
    # Check if directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"  ❌ Directory '{INPUT_DIR}' does not exist")
        return
    
    # Find all CSV files
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not input_files:
        print(f"  ⚠ No CSV files found in {INPUT_DIR}/")
        return
    
    print(f"Found {len(input_files)} file(s):\n")
    
    # Validate each file
    valid_count = 0
    total_rows = 0
    file_row_counts = []
    for filepath in input_files:
        is_valid, row_count = validate_file(filepath)
        if is_valid:
            valid_count += 1
        total_rows += row_count
        file_row_counts.append((os.path.basename(filepath), row_count))
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"  {valid_count}/{len(input_files)} files passed validation")
    
    print(f"\n  Row counts:")
    for filename, count in file_row_counts:
        print(f"    {filename}: {count} rows")
    print(f"    {'-'*30}")
    print(f"    Total: {total_rows} rows")
    
    if valid_count == len(input_files):
        print(f"\n  ✓ All files are ready for processing")
    else:
        print(f"\n  ❌ Please fix errors before running process_data.py")


if __name__ == "__main__":
    main()
