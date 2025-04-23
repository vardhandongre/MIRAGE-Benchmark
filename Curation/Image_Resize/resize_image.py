from datasets import load_dataset
from PIL import Image
import os
import json

data_path = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/workspace/B_Type/Datasets/raw_data/multi_turn_qa_data.json"

ds = load_dataset("json", data_files=data_path, split="train")

IMAGE_SAVE_DIR = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/datasets/Ask_Extension/resized_images"
ORIGINAL_IMAGE_DIR = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/datasets/Ask_Extension/images"
target_size = 672

def resize_image(example):
    attachments = example.get("attachments", [])
    new_attachments = []
    complete = True
    for image_path in attachments:
        image_path = os.path.join(ORIGINAL_IMAGE_DIR, image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        suffix = os.path.splitext(os.path.basename(image_path))[1]
        if suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            print(f"Unsupported image format: {suffix}")
            complete = False
            continue
        resized_image_name = f"{base_name}.jpg"
        resized_image_path = os.path.join(IMAGE_SAVE_DIR, resized_image_name)
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                if h > w:
                    new_h = target_size
                    new_w = int(target_size * w / h)
                else:
                    new_w = target_size
                    new_h = int(target_size * h / w)

                img_resized = img.resize((new_w, new_h), Image.BILINEAR)
                img_resized.save(resized_image_path,
                                format="JPEG",
                                optimize=True,
                                quality=95)

            new_attachments.append(resized_image_path)
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            complete = False
            continue
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


new_ds = ds.map(resize_image,num_proc=20)
new_ds = new_ds.filter(lambda x: x["complete"])
new_ds = new_ds.remove_columns(["complete"])

new_path = os.path.splitext(data_path)[0] + "_resized.json"
save_to_json_file(new_ds, new_path)