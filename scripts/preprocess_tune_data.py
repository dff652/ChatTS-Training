import json
import os
import csv
import sys

def preprocess_chatts_tune(base_dir, output_path):
    json_dir = os.path.join(base_dir, "data/chatts_tune/json")
    ts_root_dir = os.path.join(base_dir, "data/chatts_tune/timeseries")
    
    # Mapping JSON files to their TS directories
    # Mapping based on observation:
    dataset_mapping = {
        "gdsh.json": "gdsh",
        "hbsn.json": "hbsn",
        "whlj_ljsj.json": "whlj",
        "zhlh.json": "zhlh"
    }
    
    processed_data = []
    total_json_samples = 0
    total_success = 0

    for json_file, ts_subdir in dataset_mapping.items():
        json_path = os.path.join(json_dir, json_file)
        ts_dir = os.path.join(ts_root_dir, ts_subdir)
        
        if not os.path.exists(json_path):
            print(f"Skipping {json_file}: JSON not found.")
            continue
            
        print(f"Processing {json_file} using TS dir {ts_subdir}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_json_samples += len(data)
        file_success = 0
        missing_files = []

        for item in data:
            image_path = item.get('image', '')
            if not image_path:
                continue
                
            basename = os.path.basename(image_path)
            # Remove extension for cleaner handling
            base_no_ext = os.path.splitext(os.path.splitext(basename)[0])[0] # Handles .PV.jpg -> .PV -> "" logic if complex, but simple replace is safer
            
            # Original csv name from json
            csv_name_from_json = basename.replace('.jpg', '.csv')

            # Dataset-specific filename transformation rules
            potential_filenames = []
            
            # 1. Exact match (Standard)
            potential_filenames.append(csv_name_from_json)
            
            # 2. "数据集" prefix (Found in gdsh)
            potential_filenames.append("数据集" + csv_name_from_json)

            if ts_subdir == "hbsn":
                # Rule: Remove "gdsh_second_" prefix if present
                # JSON: gdsh_second_AT_032143.PV.jpg -> Disk: AT_032143.PV.csv
                if csv_name_from_json.startswith("gdsh_second_"):
                    clean_name = csv_name_from_json.replace("gdsh_second_", "")
                    potential_filenames.append(clean_name)
            
            if ts_subdir == "whlj":
                # Rule: Add "数据集whlj_ljsj_" prefix
                # JSON: NB.LJSJ.PT_2A234C.PV.jpg -> Disk: 数据集whlj_ljsj_NB.LJSJ.PT_2A234C.PV.csv
                potential_filenames.append("数据集whlj_ljsj_" + csv_name_from_json)
            
            # Check existence
            target_path = None
            for fname in potential_filenames:
                p = os.path.join(ts_dir, fname)
                if os.path.exists(p):
                    target_path = p
                    break
            
            if not target_path:
                missing_files.append(csv_name_from_json)
                continue

            ts_values = []
            try:
                with open(target_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if 'value' in row:
                            try:
                                ts_values.append(float(row['value']))
                            except ValueError:
                                pass
            except Exception as e:
                print(f"Error reading {target_path}: {e}")
                continue
                
            if not ts_values:
                continue

            conversations = item.get('conversations', [])
            prompt = ""
            response = ""
            
            for turn in conversations:
                if turn['from'] == 'user':
                    prompt = turn['value'].replace('<image>', '<ts><ts/>')
                elif turn['from'] == 'assistant':
                    response = turn['value']

            if not prompt or not response:
                continue

            processed_data.append({
                "input": prompt,
                "output": response,
                "timeseries": [ts_values]
            })
            file_success += 1
            total_success += 1
            
        print(f"  Result: {file_success}/{len(data)} successfully processed.")
        if missing_files:
            print(f"  Missing {len(missing_files)} files in {ts_subdir} (first 3): {missing_files[:3]}")

    # Save all to one JSONL
    print(f"\nFinal Summary: {total_success}/{total_json_samples} total samples processed.")
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    print("Pre-processing complete.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_jsonl = os.path.join(project_root, "data/chatts_tune/train.jsonl")
    preprocess_chatts_tune(project_root, target_jsonl)
