import sys
import json
import argparse
import os
import string
from gem_metrics import texts, compute
import multiprocessing
from tqdm import tqdm
import torch
import gc

def worker_init():
    """
    每个子进程启动时调用：
      1. 设置 TRANSFORMERS_OFFLINE=1，保证只使用本地缓存；
      2. 预加载一次模型配置，避免后续重复请求 Hugging Face 服务器。
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("distilbert-base-uncased")
        print("Worker initialized: model config loaded from cache.")
    except Exception as e:
        print("Worker initialization warning:", e)

class Evaluator:
    def __init__(self, input_file, output_file, expert_name="expert", subject_name="model", batch_size=100, num_processes=None):
        self.input_file = input_file
        self.output_file = output_file
        self.expert_name = expert_name
        self.subject_name = subject_name
        self.batch_size = batch_size
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.PUNCTUATION = set(string.punctuation)
        
    def load_data(self):
        """Load data from the input file and return the parsed JSON."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_answers(self, item):
        """Extract expert (reference) and model (prediction) answers from an item."""
        # Expert answer
        if self.expert_name not in item:
            if "b_type" in item and self.expert_name in item["b_type"]:
                expert_answer = item["b_type"][self.expert_name]
            else:
                raise ValueError(f"Expert answer not found in item {item.get('id', 'unknown')}")
        else:
            expert_answer = item[self.expert_name]
        
        # Model answer
        if self.subject_name not in item:
            if "a_type" in item and self.subject_name in item["a_type"]:
                model_response = item["a_type"][self.subject_name]
            elif "b_type" in item and self.subject_name in item["b_type"]:
                model_response = item["b_type"][self.subject_name]
            else:
                raise ValueError(f"Model response not found in item {item.get('id', 'unknown')}")
        else:
            model_response = item[self.subject_name]
        
        return expert_answer, model_response
    
    def process_batch(self, batch):
        """
        Process a batch of items:
          1. Extract predictions and references;
          2. Call gem_metrics.compute to compute BERTScore (with metrics_list=['bertscore']);
          3. Return a tuple (batch_result, count) where batch_result contains the averaged BERTScore.
          4. 清理显存，防止GPU内存不断累积。
        """
        batch_preds = []
        batch_refs = []
        for item in batch:
            try:
                ref, pred = self.extract_answers(item)
                if pred is None or ref is None:
                    print(f"Skipping item {item.get('id', 'unknown')} due to missing prediction or reference.")
                    continue
                batch_refs.append(ref)
                batch_preds.append(pred)
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
        if not batch_preds:
            return ({"precision": 0.0, "recall": 0.0, "f1": 0.0}, 0)
        preds_obj = texts.Predictions(batch_preds)
        refs_obj = texts.References(batch_refs)
        result = compute(preds_obj, refs_obj, metrics_list=['bertscore'])
        if "bertscore" in result:
            batch_result = result["bertscore"]
        else:
            batch_result = result

        # 清理GPU内存：释放未使用的显存
        torch.cuda.empty_cache()
        gc.collect()
        return (batch_result, len(batch_preds))
    
    def aggregate_results(self, results_list):
        """
        Aggregate the results from all batches with weighted average:
          - Each batch's BERTScore (precision, recall, f1) is weighted by the number of samples in that batch.
          - The final output format remains consistent with the original code.
        """
        total_count = 0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_f1 = 0.0
        for batch_result, count in results_list:
            if count == 0:
                continue
            total_count += count
            sum_precision += batch_result["precision"] * count
            sum_recall += batch_result["recall"] * count
            sum_f1 += batch_result["f1"] * count
        if total_count == 0:
            final = {
                "predictions_file": None,
                "N": 0,
                "references_file": None,
                "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            }
            return final
        final = {
            "predictions_file": None,
            "N": total_count,
            "references_file": None,
            "bertscore": {
                "precision": sum_precision / total_count,
                "recall": sum_recall / total_count,
                "f1": sum_f1 / total_count
            }
        }
        return final
    
    def evaluate(self):
        """
        Evaluate all items by:
          1. Loading data and splitting into batches;
          2. Using multiprocessing to compute BERTScore for each batch;
          3. Aggregating the batch results and writing the final result.
        """
        data = self.load_data()
        print(f"Loaded {len(data)} items for evaluation.")
        
        batches = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        pool = multiprocessing.Pool(processes=self.num_processes, initializer=worker_init)
        results_list = []
        for batch_result in tqdm(pool.imap_unordered(self.process_batch, batches), total=len(batches), desc="Evaluating batches"):
            results_list.append(batch_result)
        pool.close()
        pool.join()
        
        final_result = self.aggregate_results(results_list)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation completed. Final result written to {self.output_file}.")
        return final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model responses against expert answers with BERTScore using multi-process batch evaluation."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file for final result.")
    parser.add_argument("--expert_name", type=str, required=True, help="Field name for the expert's answer (reference).")
    parser.add_argument("--subject_name", type=str, required=True, help="Field name for the subject's answer (prediction).")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing items in each process.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()
    
    evaluator = Evaluator(
        input_file=args.input_file,
        output_file=args.output_file,
        expert_name=args.expert_name,
        subject_name=args.subject_name,
        batch_size=args.batch_size,
        num_processes=args.num_processes
    )
    
    final_result = evaluator.evaluate()
    print("Final BERTScore result:")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
