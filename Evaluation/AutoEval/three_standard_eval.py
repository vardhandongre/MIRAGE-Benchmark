import sys
import json
import argparse
import os
import multiprocessing
from tqdm import tqdm
import numpy as np
import string
from gem_metrics import texts, compute
from gem_metrics.tokenize import default_tokenize_func
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycountry import languages

class Evaluator:
    def __init__(self, input_file, output_file, expert_name="expert", subject_name="model", metrics_list=None, num_processes=None):
        self.input_file = input_file
        self.output_file = output_file
        self.expert_name = expert_name
        self.subject_name = subject_name
        # Only use metrics: bleu, rouge, and meteor.
        self.metrics_list = metrics_list or ['bleu', 'rouge', 'meteor']
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.PUNCTUATION = set(string.punctuation)
        
    def load_data(self):
        """Load data from input file and return the parsed JSON."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_answers(self, item):
        """Extract expert (reference) and model (prediction) answers from an item."""
        # Get expert answer
        if self.expert_name not in item:
            if "b_type" in item and self.expert_name in item["b_type"]:
                expert_answer = item["b_type"][self.expert_name]
            else:
                raise ValueError(f"Expert answer not found in item {item.get('id', 'unknown')}")
        else:
            expert_answer = item[self.expert_name]
        
        # Get model answer
        if self.subject_name not in item:
            if "a_type" in item and self.subject_name in item["a_type"]:
                model_response = item["a_type"][self.subject_name]
            elif "b_type" in item and self.subject_name in item["b_type"]:
                model_response = item["b_type"][self.subject_name]
            else:
                if isinstance(item.get(self.subject_name, {}), dict):
                    model_response = item[self.subject_name].get(self.model_name)
                    if model_response is None:
                        raise ValueError(f"Model name '{self.model_name}' not found in field '{self.subject_name}' of item: {item}")
                else:
                    raise ValueError(f"Model response not found in item {item.get('id', 'unknown')}")
        else:
            if isinstance(item[self.subject_name], dict):
                model_response = item[self.subject_name].get(self.model_name)
                if model_response is None:
                    raise ValueError(f"Model name '{self.model_name}' not found in field '{self.subject_name}' of item: {item}")
            else:
                model_response = item[self.subject_name]
        
        return expert_answer, model_response
    
    def meteor_score(self, gts, res):
        """Calculate METEOR score using pycocoevalcap."""
        scorer = Meteor()
        score, _ = scorer.compute_score(gts, res)
        return score
    
    def rouge_score(self, gts, res):
        """Calculate ROUGE score using pycocoevalcap."""
        scorer = Rouge()
        score, _ = scorer.compute_score(gts, res)
        return score
    
    def prepare_for_coco_metrics(self, predictions, references):
        """Tokenize texts and format them for COCO metrics calculation."""
        token_func = default_tokenize_func(languages.get(alpha_2="en"))
        
        # Tokenize predictions and references
        tokenized_predictions = [token_func(pred) for pred in predictions]
        tokenized_references = [token_func(ref) for ref in references]
        
        # Lowercase tokens
        tokenized_predictions = [[w.lower() for w in pred] for pred in tokenized_predictions]
        tokenized_references = [[w.lower() for w in ref] for ref in tokenized_references]
        
        # Remove punctuation from tokens
        processed_predictions = []
        for pred in tokenized_predictions:
            tmp_pred = [w for w in pred if w not in self.PUNCTUATION]
            processed_predictions.append(" ".join(tmp_pred))
            
        processed_references = []
        for ref in tokenized_references:
            tmp_ref = [w for w in ref if w not in self.PUNCTUATION]
            processed_references.append(" ".join(tmp_ref))
        
        # Format for COCO metrics
        gts = {i: [ref] for i, ref in enumerate(processed_references)}
        res = {i: [pred] for i, pred in enumerate(processed_predictions)}
        
        return gts, res
    
    def compute_coco_metrics(self, predictions, references):
        """Compute METEOR and ROUGE-L scores using pycocoevalcap."""
        gts, res = self.prepare_for_coco_metrics(predictions, references)
        results = {}
        results['meteor'] = self.meteor_score(gts, res)
        rouge_result = self.rouge_score(gts, res)
        results['rougeL'] = rouge_result
        return results
    
    def compute_gem_metrics(self, predictions, references):
        """Compute BLEU score using gem_metrics."""
        preds = texts.Predictions(predictions)
        refs = texts.References(references)
        # Use only BLEU in gem_metrics
        gem_metrics_list = ['bleu']
        if not gem_metrics_list:
            return {}
        result = compute(preds, refs, metrics_list=gem_metrics_list)
        return result
    
    def process_item(self, item):
        """Process a single item and compute evaluation metrics."""
        try:
            expert_answer, model_response = self.extract_answers(item)
            preds_list = [model_response]
            refs_list = [expert_answer]
            
            # Compute BLEU score via gem_metrics
            gem_results = self.compute_gem_metrics(preds_list, refs_list)
            # Compute METEOR and ROUGE-L via pycocoevalcap
            coco_results = self.compute_coco_metrics(preds_list, refs_list)
            
            # Merge results
            result = {**gem_results, **coco_results}
            result['id'] = item.get('id', 'unknown')
            return result
            
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            return None
    
    def process_item_and_save(self, args):
        """
        Wrapper for processing an item and writing the result immediately to the output file.
        This helps to continuously save intermediate results.
        """
        item, output_file, lock = args
        result = self.process_item(item)
        if result is not None:
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result.get('id') if result else None
    
    def aggregate_results(self, all_results):
        """Aggregate individual results to compute average scores."""
        valid_results = [r for r in all_results if r is not None]
        if not valid_results:
            raise ValueError("No valid results to aggregate")
        
        aggregated = {}
        for metric in ['meteor', 'bleu', 'rougeL']:
            if metric in valid_results[0]:
                aggregated[metric] = np.mean([r[metric] for r in valid_results])
        aggregated['num_samples'] = len(valid_results)
        return aggregated
    
    def cleanup_output(self, total_count):
        """
        Clean up the output file by rewriting only valid items.
        This helps remove any duplicate or partially written results.
        """
        valid_items = []
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Consider an item valid if it contains all expected metric keys
                    if "meteor" in item and "bleu" in item and "rougeL" in item:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Total successful items: {len(valid_items)}. Remaining items to process: {total_count - len(valid_items)}.")
    
    def evaluate(self):
        """Main evaluation method."""
        # Load all data
        data = self.load_data()
        total_count = len(data)
        print(f"Loaded {total_count} items for evaluation.")
        
        # Check output file to determine which items have already been processed
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        if "id" in result:
                            processed_ids.add(result["id"])
                    except json.JSONDecodeError:
                        continue
        
        # Filter out items that have already been processed
        items_to_process = [item for item in data if item.get("id") not in processed_ids]
        print(f"Processing {len(items_to_process)} items out of {total_count}.")
        
        # Set up a multiprocessing lock and pool
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        pool = multiprocessing.Pool(processes=self.num_processes)
        args_list = [(item, self.output_file, lock) for item in items_to_process]
        
        # Process items in parallel with a progress bar; each result is saved immediately
        for _ in tqdm(pool.imap_unordered(self.process_item_and_save, args_list), total=len(args_list), desc="Evaluating items"):
            pass
        pool.close()
        pool.join()
        
        # Clean up the output file to remove any duplicate or invalid entries
        self.cleanup_output(total_count)
        
        # Read all valid results from the output file for aggregation
        results = []
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    results.append(item)
                except json.JSONDecodeError:
                    continue
        
        # Aggregate results from valid items
        # aggregated_results = self.aggregate_results(results)
        print(f"Evaluation completed. Processed {len(results)} items out of {total_count}.")
        # print(f"Aggregated Results: {json.dumps(aggregated_results, indent=2, ensure_ascii=False)}")
        # return aggregated_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model responses against expert answers.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file for individual results.")
    parser.add_argument("--expert_name", type=str, required=True, help="Name of the expert's answer field (used as reference).")
    parser.add_argument("--subject_name", type=str, required=True, help="Name of the subject's answer field (used as prediction).")
    parser.add_argument("--metrics", type=str, default="bleu,rouge,meteor", help="Comma-separated list of metrics to compute.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    metrics_list = args.metrics.split(',')
    
    evaluator = Evaluator(
        input_file=args.input_file,
        output_file=args.output_file,
        expert_name=args.expert_name,
        subject_name=args.subject_name,
        metrics_list=metrics_list,
        num_processes=args.num_processes
    )
    
    results = evaluator.evaluate()
    print("Aggregated Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
