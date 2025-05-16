import os
import json
import argparse
from tqdm import tqdm

def combine_batches(input_dir: str, output_file: str):
    combined = []
    files = sorted(f for f in os.listdir(input_dir) if f.startswith("batch_") and f.endswith(".json"))

    print(f"🔍 Found {len(files)} batch files in {input_dir}")
    for fname in tqdm(files, desc="Combining batches"):
        path = os.path.join(input_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
                combined.extend(data)
        except Exception as e:
            print(f"⚠️ Failed to read {fname}: {e}")

    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"✅ Combined {len(combined)} entries into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine JSON batches into one file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with batch_*.json files", default="../data/batches")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output combined JSON file", default="../data/all_generated_tasks.json")
    args = parser.parse_args()

    combine_batches(args.input_dir, args.output_file)
