import csv
import os
import glob

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory
INPUT_DIR = os.path.join(SCRIPT_DIR, "to-label")

# Typo mappings
IN_TYPOS = ["in ", "oin", " i", "i", "tin"]
OUT_TYPOS = ["pout", "tout", "otu", "ub", "ou ", "ou", "iout", "c  c"]


def fix_anamt3_columns(filepath):
    """Special fix for anamt3.csv - reorder columns."""
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Reorder: move col 2,3 to 1,2 and orig col 1 to col 3
    # Original: [col0, col1, col2, col3] -> [col0, col2, col3, col1]
    for i in range(len(rows)):
        orig_col1 = rows[i][1]
        orig_col2 = rows[i][2]
        orig_col3 = rows[i][3]
        rows[i][1] = orig_col2  # col 2 -> col 1
        rows[i][2] = orig_col3  # col 3 -> col 2
        rows[i][3] = orig_col1  # orig col 1 -> col 3
    
    with open(filepath, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    
    print(f"  anamt3.csv: Reordered columns [0,1,2,3] -> [0,2,3,1]")
    return 1


def fix_file(filepath):
    """Fix label typos in a single CSV file."""
    filename = os.path.basename(filepath)
    
    # Special handling for anamt3.csv - reorder columns first
    # if filename == 'anamt3.csv':
    #     fix_anamt3_columns(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    fixes_made = 0
    
    # Strip trailing empty columns
    if len(rows[0]) > 4:
        has_trailing_empty = all(col.strip() == '' for col in rows[0][4:])
        if has_trailing_empty:
            for i in range(len(rows)):
                rows[i] = rows[i][:4]
            fixes_made += 1
            print(f"  {filename}: Removed trailing empty columns")
    
    # Fix header typos
    if rows[0][0] == 'in/out/ambigious':
        rows[0][0] = 'in/out/ambiguous'
        fixes_made += 1
        print(f"  {filename} header: 'in/out/ambigious' -> 'in/out/ambiguous'")
    
    if rows[0][0] == 'in/out/ambigous':
        rows[0][0] = 'in/out/ambiguous'
        fixes_made += 1
        print(f"  {filename} header: 'in/out/ambigous' -> 'in/out/ambiguous'")
    
    if rows[0][0] == 'in/out/ambiguos':
        rows[0][0] = 'in/out/ambiguous'
        fixes_made += 1
        print(f"  {filename} header: 'in/out/ambiguos' -> 'in/out/ambiguous'")
    
    if rows[0][1] == 'Start':
        rows[0][1] = 'start'
        fixes_made += 1
        print(f"  {filename} header: 'Start' -> 'start'")
    
    if rows[0][2] == 'End':
        rows[0][2] = 'end'
        fixes_made += 1
        print(f"  {filename} header: 'End' -> 'end'")
    
    if len(rows[0]) > 3 and rows[0][3] == 'Trancript':
        rows[0][3] = 'Transcript'
        fixes_made += 1
        print(f"  {filename} header: 'Trancript' -> 'Transcript'")
    
    if len(rows[0]) > 3 and rows[0][3] == 'Transicript':
        rows[0][3] = 'Transcript'
        fixes_made += 1
        print(f"  {filename} header: 'Transicript' -> 'Transcript'")
    
    # Handle alternate header format: ['in/out/ambiguous', 'Transcript', 'Final_Beginning', 'Final_End']
    # Needs to become: ['in/out/ambiguous', 'start', 'end', 'Transcript']
    if (len(rows[0]) == 4 and 
        rows[0][1] == 'Transcript' and 
        rows[0][2] == 'Final_Beginning' and 
        rows[0][3] == 'Final_End'):
        
        # Reorder all rows: [col0, col1, col2, col3] -> [col0, col2, col3, col1]
        for i in range(len(rows)):
            orig_col1 = rows[i][1]  # Transcript
            orig_col2 = rows[i][2]  # Final_Beginning / start value
            orig_col3 = rows[i][3]  # Final_End / end value
            rows[i][1] = orig_col2
            rows[i][2] = orig_col3
            rows[i][3] = orig_col1
        
        # Rename headers
        rows[0][1] = 'start'
        rows[0][2] = 'end'
        rows[0][3] = 'Transcript'
        
        fixes_made += 1
        print(f"  {filename}: Reordered columns and renamed headers (4-col Final_Beginning/Final_End format)")
    
    # Handle 8-column format: ['in/out/ambiguous', 'Transcript', 'JAS_Beginning', 'JAS_End', 'JOC_Beginning', 'JOC_End', 'Final_Beginnin', 'Final_End']
    # Extract only: col0, col6 (start), col7 (end), col1 (Transcript)
    if (len(rows[0]) == 8 and 
        rows[0][1] == 'Transcript' and
        ('Final_Beginnin' in rows[0][6] or 'Final_Beginning' in rows[0][6])):
        
        # Rebuild rows with only 4 columns
        new_rows = []
        for i, row in enumerate(rows):
            new_row = [
                row[0],  # in/out/ambiguous
                row[6],  # Final_Beginnin -> start
                row[7],  # Final_End -> end
                row[1]   # Transcript
            ]
            new_rows.append(new_row)
        
        # Replace rows
        rows.clear()
        rows.extend(new_rows)
        
        # Fix header names
        rows[0][1] = 'start'
        rows[0][2] = 'end'
        rows[0][3] = 'Transcript'
        
        fixes_made += 1
        print(f"  {filename}: Extracted 4 columns from 8-column format")
    
    # Skip header row
    for i in range(1, len(rows)):
        # Fix in/out/ambiguous column (column 0)
        original = rows[i][0]
        
        # Check for "in" typos
        if original in IN_TYPOS:
            rows[i][0] = "in"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 0 '{original}' -> 'in'")
        
        # Check for "out" typos
        elif original in OUT_TYPOS:
            rows[i][0] = "out"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 0 '{original}' -> 'out'")
        
        # Check for empty in/out column - default to "out"
        elif original.strip() == "":
            rows[i][0] = "out"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 0 empty -> 'out'")
        
        # Fix start column (column 1)
        if rows[i][1].strip() == "":
            rows[i][1] = "FALSE"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 1 (start) empty -> 'FALSE'")
        elif rows[i][1].strip() == "RUE":
            rows[i][1] = "TRUE"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 1 (start) 'RUE' -> 'TRUE'")
        elif rows[i][1].strip() == "TRUETRUE":
            rows[i][1] = "TRUE"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 1 (start) 'TRUETRUE' -> 'TRUE'")
        
        # Fix end column (column 2)
        if rows[i][2].strip() == "":
            rows[i][2] = "FALSE"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 2 (end) empty -> 'FALSE'")
        elif rows[i][2].strip() == "RUE":
            rows[i][2] = "TRUE"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 2 (end) 'RUE' -> 'TRUE'")
        elif rows[i][2].strip() == "TRUETRUE":
            rows[i][2] = "TRUE"
            fixes_made += 1
            print(f"  {filename} line {i+1}: col 2 (end) 'TRUETRUE' -> 'TRUE'")
    
    # Write back if fixes were made
    if fixes_made > 0:
        with open(filepath, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
    
    return fixes_made


def main():
    print(f"{'='*60}")
    print("FIX LABEL TYPOS")
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
    
    # Fix each file
    total_fixes = 0
    for filepath in input_files:
        fixes = fix_file(filepath)
        total_fixes += fixes
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    if total_fixes > 0:
        print(f"  ✓ Fixed {total_fixes} typo(s)")
    else:
        print(f"  ✓ No typos found")


if __name__ == "__main__":
    main()
