import csv
import os
import glob

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories (relative to script location)
HUMAN_LABELED_DIR = os.path.join(SCRIPT_DIR, "to-label")        # Original human-labeled files
LLM_LABELED_DIR = os.path.join(SCRIPT_DIR, "labeled-out")       # LLM-labeled files from process_data.py
COMPARE_DIR = os.path.join(SCRIPT_DIR, "labeled-compare")       # Comparison output files


def analyze_transcript(human_path, llm_path, compare_path):
    """Compare human and LLM labels for a single transcript."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(human_path)}")
    print(f"  Human labels: {human_path}")
    print(f"  LLM labels: {llm_path}")
    print(f"  Output: {compare_path}")
    print(f"{'='*60}")
    
    # Read the human-labeled data
    with open(human_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        human_rows = list(reader)
    
    # Read the LLM-generated data
    with open(llm_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        llm_rows = list(reader)
    
    # Create the comparison file
    with open(compare_path, 'w', encoding='utf-8', newline='') as file:
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
    with open(compare_path, 'r', encoding='utf-8') as file:
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

    # =========================================
    # 1. START DETECTION METRICS
    # =========================================
    start_tp = 0  # Both say start=TRUE
    start_tn = 0  # Both say start=FALSE
    start_fp = 0  # LLM says TRUE, Human says FALSE
    start_fn = 0  # LLM says FALSE, Human says TRUE
    
    index = 1
    while index <= run_length:
        human_start = rows[index][1] == 'TRUE'
        llm_start = rows[index][2] == 'TRUE'
        
        if human_start and llm_start:
            start_tp += 1
        elif not human_start and not llm_start:
            start_tn += 1
        elif llm_start and not human_start:
            start_fp += 1
        elif not llm_start and human_start:
            start_fn += 1
        index += 1
    
    # =========================================
    # 2. END DETECTION METRICS
    # =========================================
    end_tp = 0
    end_tn = 0
    end_fp = 0
    end_fn = 0
    
    index = 1
    while index <= run_length:
        human_end = rows[index][3] == 'TRUE'
        llm_end = rows[index][4] == 'TRUE'
        
        if human_end and llm_end:
            end_tp += 1
        elif not human_end and not llm_end:
            end_tn += 1
        elif llm_end and not human_end:
            end_fp += 1
        elif not llm_end and human_end:
            end_fn += 1
        index += 1
    
    # =========================================
    # 3. SEGMENT OVERLAP METRICS
    # =========================================
    seg_tp = 0  # Both say story segment
    seg_tn = 0  # Both say not story segment
    seg_fp = 0  # LLM says segment, Human says no
    seg_fn = 0  # LLM says no, Human says segment
    intersection_count = 0
    union_count = 0
    
    index = 1
    while index <= run_length:
        human_seg = rows[index][6] == 'TRUE'
        llm_seg = rows[index][7] == 'TRUE'
        
        # Intersection and Union
        if human_seg and llm_seg:
            rows[index][8] = 'TRUE'
            intersection_count += 1
        if human_seg or llm_seg:
            rows[index][9] = 'TRUE'
            union_count += 1
        
        # Confusion matrix for segments
        if human_seg and llm_seg:
            seg_tp += 1
        elif not human_seg and not llm_seg:
            seg_tn += 1
        elif llm_seg and not human_seg:
            seg_fp += 1
        elif not llm_seg and human_seg:
            seg_fn += 1
        
        index += 1
    
    # =========================================
    # CALCULATE ALL METRICS
    # =========================================
    def calc_metrics(tp, tn, fp, fn):
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    start_metrics = calc_metrics(start_tp, start_tn, start_fp, start_fn)
    end_metrics = calc_metrics(end_tp, end_tn, end_fp, end_fn)
    seg_metrics = calc_metrics(seg_tp, seg_tn, seg_fp, seg_fn)
    seg_metrics['iou'] = intersection_count / union_count if union_count > 0 else 0
    
    # Print metrics
    print(f"\n  START DETECTION:")
    print(f"    TP: {start_tp}, TN: {start_tn}, FP: {start_fp}, FN: {start_fn}")
    print(f"    Acc: {start_metrics['accuracy']:.4f}, Prec: {start_metrics['precision']:.4f}, Rec: {start_metrics['recall']:.4f}, F1: {start_metrics['f1']:.4f}")
    
    print(f"\n  END DETECTION:")
    print(f"    TP: {end_tp}, TN: {end_tn}, FP: {end_fp}, FN: {end_fn}")
    print(f"    Acc: {end_metrics['accuracy']:.4f}, Prec: {end_metrics['precision']:.4f}, Rec: {end_metrics['recall']:.4f}, F1: {end_metrics['f1']:.4f}")
    
    print(f"\n  SEGMENT OVERLAP:")
    print(f"    TP: {seg_tp}, TN: {seg_tn}, FP: {seg_fp}, FN: {seg_fn}")
    print(f"    Acc: {seg_metrics['accuracy']:.4f}, Prec: {seg_metrics['precision']:.4f}, Rec: {seg_metrics['recall']:.4f}, F1: {seg_metrics['f1']:.4f}, IoU: {seg_metrics['iou']:.4f}")
    
    # Write the updated rows back to the file
    with open(compare_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    
    return {
        'start': start_metrics,
        'end': end_metrics,
        'segment': seg_metrics
    }


def main():
    # Ensure output directory exists
    os.makedirs(COMPARE_DIR, exist_ok=True)
    
    # Find all CSV files in the human-labeled directory
    human_files = glob.glob(os.path.join(HUMAN_LABELED_DIR, "*.csv"))
    
    if not human_files:
        print(f"No CSV files found in {HUMAN_LABELED_DIR}/")
        return
    
    print(f"Found {len(human_files)} file(s) to analyze:")
    for f in human_files:
        print(f"  - {os.path.basename(f)}")
    
    # Track overall stats for each metric type
    totals = {
        'start': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'end': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'segment': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    }
    results = []
    
    # Process each file
    for human_path in human_files:
        filename = os.path.basename(human_path)
        name, ext = os.path.splitext(filename)
        
        # Find corresponding LLM-labeled file
        llm_filename = f"{name}_labeled{ext}"
        llm_path = os.path.join(LLM_LABELED_DIR, llm_filename)
        
        if not os.path.exists(llm_path):
            print(f"\nWarning: LLM-labeled file not found: {llm_path}")
            print(f"  Skipping {filename}")
            continue
        
        # Create comparison output path
        compare_filename = f"{name}_compare{ext}"
        compare_path = os.path.join(COMPARE_DIR, compare_filename)
        
        # Analyze this transcript
        metrics = analyze_transcript(human_path, llm_path, compare_path)
        results.append((filename, metrics))
        
        # Accumulate totals
        for metric_type in ['start', 'end', 'segment']:
            for key in ['tp', 'tn', 'fp', 'fn']:
                totals[metric_type][key] += metrics[metric_type][key]
    
    # Helper function for calculating metrics
    def calc_overall(t):
        total = t['tp'] + t['tn'] + t['fp'] + t['fn']
        acc = (t['tp'] + t['tn']) / total if total > 0 else 0
        prec = t['tp'] / (t['tp'] + t['fp']) if (t['tp'] + t['fp']) > 0 else 0
        rec = t['tp'] / (t['tp'] + t['fn']) if (t['tp'] + t['fn']) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, **t}
    
    overall_start = calc_overall(totals['start'])
    overall_end = calc_overall(totals['end'])
    overall_seg = calc_overall(totals['segment'])
    # IoU for segment
    seg_t = totals['segment']
    overall_seg['iou'] = seg_t['tp'] / (seg_t['tp'] + seg_t['fp'] + seg_t['fn']) if (seg_t['tp'] + seg_t['fp'] + seg_t['fn']) > 0 else 0
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - START DETECTION")
    print(f"{'='*80}")
    print(f"  {'File':<20} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for filename, metrics in results:
        m = metrics['start']
        print(f"  {filename:<20} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY - END DETECTION")
    print(f"{'='*80}")
    print(f"  {'File':<20} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for filename, metrics in results:
        m = metrics['end']
        print(f"  {filename:<20} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY - SEGMENT OVERLAP")
    print(f"{'='*80}")
    print(f"  {'File':<20} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'IoU':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for filename, metrics in results:
        m = metrics['segment']
        print(f"  {filename:<20} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['iou']:>8.4f}")
    
    print(f"\n{'='*80}")
    print("OVERALL METRICS")
    print(f"{'='*80}")
    
    total_samples = totals['segment']['tp'] + totals['segment']['tn'] + totals['segment']['fp'] + totals['segment']['fn']
    print(f"  Total samples: {total_samples}")
    
    print(f"\n  START DETECTION:")
    print(f"    TP: {totals['start']['tp']}, TN: {totals['start']['tn']}, FP: {totals['start']['fp']}, FN: {totals['start']['fn']}")
    print(f"    Accuracy: {overall_start['accuracy']:.4f}, Precision: {overall_start['precision']:.4f}, Recall: {overall_start['recall']:.4f}, F1: {overall_start['f1']:.4f}")
    
    print(f"\n  END DETECTION:")
    print(f"    TP: {totals['end']['tp']}, TN: {totals['end']['tn']}, FP: {totals['end']['fp']}, FN: {totals['end']['fn']}")
    print(f"    Accuracy: {overall_end['accuracy']:.4f}, Precision: {overall_end['precision']:.4f}, Recall: {overall_end['recall']:.4f}, F1: {overall_end['f1']:.4f}")
    
    print(f"\n  SEGMENT OVERLAP:")
    print(f"    TP: {totals['segment']['tp']}, TN: {totals['segment']['tn']}, FP: {totals['segment']['fp']}, FN: {totals['segment']['fn']}")
    print(f"    Accuracy: {overall_seg['accuracy']:.4f}, Precision: {overall_seg['precision']:.4f}, Recall: {overall_seg['recall']:.4f}, F1: {overall_seg['f1']:.4f}, IoU: {overall_seg['iou']:.4f}")
    
    print(f"\nComparison files saved to {COMPARE_DIR}/")


if __name__ == "__main__":
    main()
