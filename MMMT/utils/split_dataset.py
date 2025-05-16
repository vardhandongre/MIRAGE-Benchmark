import json
import random
import argparse
from pathlib import Path
import os

def split_dataset(input_path, train_frac=0.7, dev_frac=0.15, seed=42):
    random.seed(seed)
    with open(input_path) as f:
        data = json.load(f)

    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_frac)
    dev_end = train_end + int(total * dev_frac)

    train = data[:train_end]
    dev = data[train_end:dev_end]
    test = data[dev_end:]

    return train, dev, test

def write_split(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def extract_source_ids(data):
    return {item["source_id"] for item in data}

def split_source_dialogs(source_path, train_ids, dev_ids):
    with open(source_path) as f:
        all_dialogs = json.load(f)

    train = [d for d in all_dialogs if d["id"] in train_ids]
    dev = [d for d in all_dialogs if d["id"] in dev_ids]
    return train, dev

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help="Path to generated task file")
    parser.add_argument("--source", required=True, help="Path to original multi-turn QA JSON")
    parser.add_argument("--outdir", default="data/splits", help="Output directory for splits")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train, dev, test = split_dataset(args.generated)

    write_split(train, f"{args.outdir}/train.json")
    write_split(dev, f"{args.outdir}/dev.json")
    write_split(test, f"{args.outdir}/test.json")

    train_ids = extract_source_ids(train)
    dev_ids = extract_source_ids(dev)
    train_raw, dev_raw = split_source_dialogs(args.source, train_ids, dev_ids)

    write_split(train_raw, f"{args.outdir}/source_train.json")
    write_split(dev_raw, f"{args.outdir}/source_dev.json")

    print("✅ Splits written:")
    print(f" - train.json: {len(train)}")
    print(f" - dev.json:   {len(dev)}")
    print(f" - test.json:  {len(test)} (no source dialog included)")