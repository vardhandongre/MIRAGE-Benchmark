import sys
import os
import json
import argparse
import multiprocessing
from tqdm import tqdm
import numpy as np
import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from packaging import version
import sklearn.preprocessing
import collections
import warnings
import string

# ---------------------------
# Original Data Processing Code
# ---------------------------

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        # The dataset automatically adds the prefix to each caption.
        self.prefix = prefix if prefix.endswith(' ') else prefix + ' '

    def __getitem__(self, idx):
        caption = self.data[idx]
        tokens = clip.tokenize(self.prefix + caption, truncate=True).squeeze()
        return {'caption': tokens}

    def __len__(self):
        return len(self.data)

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # Only 224x224 ViT-B/32 supported for now.
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)

def extract_all_captions(captions, model, device, batch_size=256, num_workers=0):
    data_loader = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['caption'].to(device)
            all_text_features.append(model.encode_text(tokens).cpu().numpy())
    return np.vstack(all_text_features)

def extract_all_images(images, model, device, batch_size=64, num_workers=0):
    data_loader = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch['image'].to(device)
            if device == 'cuda':
                imgs = imgs.to(torch.float16)
            all_image_features.append(model.encode_image(imgs).cpu().numpy())
    return np.vstack(all_image_features)

def get_clip_score(model, images, candidates, device, w=2.5):
    """
    Compute standard image-text CLIPScore.
    images can be either a list of image paths or a precomputed numpy array.
    """
    if isinstance(images, list):
        images = extract_all_images(images, model, device)
    candidates = extract_all_captions(candidates, model, device)
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'Due to numerical instability, new numpy normalization differs from paper results. '
            'To exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
    per_instance = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per_instance), per_instance, candidates

def get_refonlyclipscore(model, references, candidates, device):
    """
    Compute the text-text similarity for RefCLIPScore.
    references: a list (one per candidate) of lists of reference captions.
    candidates: can be either a list of candidate captions or precomputed features.
    """
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)
    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx] * len(refs))
    flattened_refs = extract_all_captions(flattened_refs, model, device)
    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
        flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis=1)
    else:
        warnings.warn(
            'Due to numerical instability, new numpy normalization differs from paper results. '
            'To exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
        flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))
    cand_idx2refs = collections.defaultdict(list)
    for ref_feat, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feat)
    assert len(cand_idx2refs) == len(candidates)
    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}
    per_instance = []
    for c_idx, cand in enumerate(candidates):
        cur_refs = cand_idx2refs[c_idx]
        sims = cand.dot(cur_refs.transpose())
        per_instance.append(np.max(sims))
    return np.mean(per_instance), per_instance

# ---------------------------
# Evaluator Using Multiprocessing
# ---------------------------

class ClipEvaluator:
    def __init__(self, input_file, output_file, expert_name="refs", subject_name="candidate", num_processes=None):
        """
        Initialize evaluator with:
        - input_file: JSON file containing a list of items.
        - output_file: JSONL file where each result is appended.
        - expert_name: Field name for reference captions.
        - subject_name: Field name for candidate captions.
        - num_processes: Number of processes for parallel processing.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.expert_name = expert_name  # Reference field
        self.subject_name = subject_name  # Candidate field
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.w = 2.5  # Weight factor used in CLIPScore computation.
        # This prefix is used in the datasets.
        self.caption_prefix = "A photo depicts"
        if not self.caption_prefix.endswith(" "):
            self.caption_prefix += " "

    def load_data(self):
        """Load and return data from the input JSON file."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_item_data(self, item):
        """
        Extract candidate caption, reference caption(s), and image path from an item.
        - Candidate caption is taken from the field self.subject_name.
        - Reference caption(s) is taken from the field self.expert_name.
          If a single string is provided, it is wrapped as a list.
          Finally, wrap the references in a list (one per candidate) as expected by get_refonlyclipscore.
        - The image path is assumed to be the first element in item["attachments"].
        """
        if self.expert_name not in item:
            if "b_type" in item and self.expert_name in item["b_type"]:
                expert_answer = item["b_type"][self.expert_name]
            else:
                raise ValueError(f"Expert answer not found in item {item.get('id', 'unknown')}")
        else:
            expert_answer = item[self.expert_name]
        refs = expert_answer
        if isinstance(refs, str):
            refs = [refs]
        # Wrap in a list of lists.
        refs = [refs]
        if self.subject_name not in item:
            if "a_type" in item and self.subject_name in item["a_type"]:
                model_response = item["a_type"][self.subject_name]
            elif "b_type" in item and self.subject_name in item["b_type"]:
                model_response = item["b_type"][self.subject_name]
            else:
                raise ValueError(f"Model response not found in item {item.get('id', 'unknown')}")
        else:
            model_response = item[self.subject_name]
        candidate = model_response
        if "attachments" not in item or not item["attachments"]:
            raise ValueError(f"No attachments found in item {item.get('id', 'unknown')}")
        image_path = item["attachments"][0]
        return refs, candidate, image_path

    def load_clip_model(self, device):
        """
        Load the CLIP model (ViT-B/32) and return it.
        This is called once per process.
        """
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        return model

    def process_item(self, item):
        """
        Process a single item:
        - Extract candidate, references, and image path.
        - Compute image-text CLIPScore using the original functions.
        - If references exist, compute text-text similarity score and combine
          via harmonic mean to produce RefCLIPScore.
        Returns a dictionary with id, CLIPScore, and RefCLIPScore.
        """
        try:
            # Determine device.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Load CLIP model if not already loaded in this process.
            if not hasattr(self, "clip_model"):
                self.clip_model = self.load_clip_model(device)
            
            refs, candidate, image_path = self.extract_item_data(item)
            # Prepare candidate and image lists (each with one element).
            candidate_list = [candidate]
            image_list = [image_path]

            # Compute image-text similarity score.
            _, per_inst_img, candidate_feats = get_clip_score(
                self.clip_model, image_list, candidate_list, device, w=self.w)
            clip_score = per_inst_img[0]

            # Compute text-text similarity score using candidate features.
            _, per_inst_text = get_refonlyclipscore(
                self.clip_model, refs, candidate_feats, device)
            text_score = per_inst_text[0]

            # Compute RefCLIPScore as the harmonic mean.
            if clip_score + text_score > 0:
                refclip_score = 2 * clip_score * text_score / (clip_score + text_score)
            else:
                refclip_score = 0.0

            result = {
                "id": item.get("id", "unknown"),
                "CLIPScore": float(clip_score),
                "RefCLIPScore": float(refclip_score)
            }
            return result
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            return None

    def process_item_and_save(self, args):
        """
        Wrapper for processing an item and immediately saving its result to the output file.
        Uses a multiprocessing lock to ensure safe file writes.
        """
        item, output_file, lock = args
        result = self.process_item(item)
        if result is not None:
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result.get("id") if result is not None else None

    def cleanup_output(self, total_count):
        """
        Clean up the output file by keeping only valid items (those with both metrics).
        This step removes duplicates or partially written entries.
        """
        valid_items = []
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res_item = json.loads(line)
                    if "CLIPScore" in res_item and "RefCLIPScore" in res_item:
                        valid_items.append(res_item)
                except json.JSONDecodeError:
                    continue
        with open(self.output_file, "w", encoding="utf-8") as f:
            for res_item in valid_items:
                f.write(json.dumps(res_item, ensure_ascii=False) + "\n")
        print(f"Total successful items: {len(valid_items)}. Remaining items to process: {total_count - len(valid_items)}.")

    def evaluate(self):
        """
        Main evaluation method:
        - Load data.
        - Check which items have already been processed (by reading the output file).
        - Process remaining items in parallel using multiprocessing.
        - Clean up the output file and aggregate results.
        """
        data = self.load_data()
        total_count = len(data)
        print(f"Loaded {total_count} items for evaluation.")

        # Determine already processed item ids.
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        res_item = json.loads(line)
                        if "id" in res_item:
                            processed_ids.add(res_item["id"])
                    except json.JSONDecodeError:
                        continue

        # Filter out items already processed.
        items_to_process = [item for item in data if item.get("id") not in processed_ids]
        print(f"Processing {len(items_to_process)} items out of {total_count}.")

        manager = multiprocessing.Manager()
        lock = manager.Lock()
        pool = multiprocessing.Pool(processes=self.num_processes)
        args_list = [(item, self.output_file, lock) for item in items_to_process]

        for _ in tqdm(pool.imap_unordered(self.process_item_and_save, args_list),
                      total=len(args_list), desc="Evaluating items"):
            pass
        pool.close()
        pool.join()

        self.cleanup_output(total_count)

        results = []
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res_item = json.loads(line)
                    results.append(res_item)
                except json.JSONDecodeError:
                    continue

        print(f"Evaluation completed. Processed {len(results)} items out of {total_count}.")
        return results

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Evaluate CLIPScore and RefCLIPScore using original data processing.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file for results.")
    parser.add_argument("--expert_name", type=str, default="refs", help="Field name for reference captions (default: refs).")
    parser.add_argument("--subject_name", type=str, default="candidate", help="Field name for candidate caption (default: candidate).")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    evaluator = ClipEvaluator(
        input_file=args.input_file,
        output_file=args.output_file,
        expert_name=args.expert_name,
        subject_name=args.subject_name,
        num_processes=args.num_processes
    )

    results = evaluator.evaluate()
    print("Aggregated Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
