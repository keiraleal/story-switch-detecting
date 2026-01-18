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
        writer.writerow(['in/out/ambiguous', 'start_human', 'start_llm', 'end_human', 'end_llm', 'Transcript'])
        
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
                human_row[3]       # Transcript
            ])
            index += 1


if __name__ == "__main__":
    main()
