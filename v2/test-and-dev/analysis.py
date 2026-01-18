import csv


def main():
    # Read the human-labeled data
    with open('trial_data.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        human_rows = list(reader)
    
    # Read the LLM-generated data
    with open('trial_data_out.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        llm_rows = list(reader)
    
    # Create the comparison file
    with open('trial_data_compare.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(['in/out/ambiguous', 'start_human', 'start_llm', 'end_human', 'end_llm', 'Transcript', 'story_seg_human', 'story_seg_llm', 'intersection', 'union'])
        
        # Write data rows
        index = 1
        while index < len(human_rows):
            human_row = human_rows[index]
            llm_row = llm_rows[index]
            
            writer.writerow([
                human_row[0],      # in/out/ambiguous
                human_row[1],      # start_human
                llm_row[1],        # start_llm
                human_row[2],      # end_human
                llm_row[2],        # end_llm
                human_row[3],      # Transcript
                'FALSE',           # story_seg_human (initialized to FALSE)
                'FALSE',           # story_seg_llm (initialized to FALSE)
                'FALSE',           # intersection (initialized to FALSE)
                'FALSE'            # union (initialized to FALSE)
            ])
            index += 1

    # Read the comparison file we just created
    with open('trial_data_compare.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    index = 1
    toggle_human = False
    toggle_llm = False
    run_length = len(rows) - 1
    while index <= run_length:
        # Human: turn on at start
        if rows[index][1] == 'TRUE':
            toggle_human = True
        # Mark row before turning off (so end row is included)
        if toggle_human:
            rows[index][6] = 'TRUE'
        # Human: turn off at end
        if rows[index][3] == 'TRUE':
            toggle_human = False
        
        # LLM: turn on at start
        if rows[index][2] == 'TRUE':
            toggle_llm = True
        # Mark row before turning off (so end row is included)
        if toggle_llm:
            rows[index][7] = 'TRUE'
        # LLM: turn off at end
        if rows[index][4] == 'TRUE':
            toggle_llm = False
        
        index += 1

    index = 1
    intersection_count = 0
    union_count = 0
    while index <= run_length:
        if rows[index][6] == 'TRUE' and rows[index][7] == 'TRUE':
            rows[index][8] = 'TRUE'
            intersection_count += 1
        if rows[index][6] == 'TRUE' or rows[index][7] == 'TRUE':
            rows[index][9] = 'TRUE'
            union_count += 1
        index += 1
    
    # Calculate and print IoU
    if union_count > 0:
        iou = intersection_count / union_count
        print(f"Intersection count: {intersection_count}")
        print(f"Union count: {union_count}")
        print(f"IoU (Intersection over Union): {iou:.4f}")
    else:
        print("No union found (no story segments detected)")
    
    # Write the updated rows back to the file
    with open('trial_data_compare.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

if __name__ == "__main__":
    main()
