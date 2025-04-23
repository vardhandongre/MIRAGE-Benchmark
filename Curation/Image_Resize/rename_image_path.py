from datasets import load_dataset
import os
import json

NEW_IMAGE_SAVE_DIR = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/datasets/Ask_Extension/resized_images"


def rename_image_path(example):
    attachments = example.get("attachments", [])
    new_attachments = []
    complete = True
    for image_path in attachments:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        suffix = os.path.splitext(os.path.basename(image_path))[1]
        if suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            print(f"Unsupported image format: {suffix}")
            complete = False
            continue
        new_image_path = os.path.join(NEW_IMAGE_SAVE_DIR, f"{base_name}.jpg")
        new_attachments.append(new_image_path)
    return {"attachments": new_attachments, "complete": complete}
import datetime

def serialize_example(ex):
    out = {}
    for k, v in ex.items():
        # 如果是 pandas.Timestamp 或 datetime.datetime，转成 ISO 字符串
        if isinstance(v, (datetime.datetime,)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out

def save_to_json_file(dataset, file_name):
    dict_dataset = [serialize_example(example) for example in dataset]
    with open(file_name, 'w') as f:
        json.dump(dict_dataset, f, indent=4)
    print(f"Saved {len(dict_dataset)} items to {file_name}")

# data_path = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/workspace/B_Type/Datasets/pre_data/valid_single_turn_links_images_contents_dataset.json"
# data = load_dataset("json", data_files=data_path, split="train")
# new_ds = data.map(rename_image_path,num_proc=20)
# new_ds = new_ds.filter(lambda x: x["complete"] == True)
# new_ds = new_ds.remove_columns(["complete"])
# new_path = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/workspace/B_Type/Datasets/pre_data/valid_single_turn_links_images_contents_dataset_resized.json"
# save_to_json_file(new_ds, new_path)


data_dir = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/workspace/B_Type/Datasets/benchmarks"
for file_name in os.listdir(data_dir):
    if not file_name.endswith(".json"):
        continue
    data_path = os.path.join(data_dir, file_name)
    data = load_dataset("json", data_files=data_path, split="train")
    new_ds = data.map(rename_image_path,num_proc=20)
    new_ds = new_ds.filter(lambda x: x["complete"] == True)
    new_ds = new_ds.remove_columns(["complete"])
    save_to_json_file(new_ds, data_path)