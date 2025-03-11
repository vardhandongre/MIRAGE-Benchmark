import sys
import json
import argparse
import os
import string
from bleurt import score  # Import BLEURT scorer

class Evaluator:
    def __init__(self, input_file, output_file, expert_name="expert", subject_name="model"):
        self.input_file = input_file
        self.output_file = output_file
        self.expert_name = expert_name
        self.subject_name = subject_name
        self.PUNCTUATION = set(string.punctuation)
        
    def load_data(self):
        """
        Load data from the input file and parse it as JSON.
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_answers(self, item):
        """
        Extract the expert (reference) and model (prediction) answers from an item.
        """
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
                raise ValueError(f"Model response not found in item {item.get('id', 'unknown')}")
        else:
            model_response = item[self.subject_name]
        
        return expert_answer, model_response
    
    def compute_bleurt(self, predictions, references):
        """
        Compute BLEURT scores using the BLEURT scorer.
        This function initializes the scorer with the specified checkpoint and computes
        the BLEURT score for all predictions against the references.
        """
        checkpoint = "/taiga/ncsa/radiant/bbgp/cropwizard/chigui/huggingface/hub/BLEURT-20"
        scorer = score.BleurtScorer(checkpoint)
        scores = scorer.score(references=references, candidates=predictions)
        return {"bleurt": scores}
    
    def evaluate(self):
        """
        Evaluate all items by:
        1. Extracting predictions and references into separate lists.
        2. Computing BLEURT scores for all predictions and references at once.
        3. Writing the final result to the output file.
        """
        # Load all data
        data = self.load_data()
        print(f"Loaded {len(data)} items for evaluation.")
        
        all_refs = []
        all_preds = []
        
        # Extract answers from each item
        for item in data:
            try:
                ref, pred = self.extract_answers(item)
                all_refs.append(ref)
                all_preds.append(pred)
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
        
        # Compute BLEURT scores for all predictions and references at once
        result = self.compute_bleurt(all_preds, all_refs)
        
        # Write the final result to the output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation completed. Final result written to {self.output_file}.")
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model responses against expert answers using a single BLEURT call."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file for the final result.")
    parser.add_argument("--expert_name", type=str, required=True, help="Field name for the expert's answer (reference).")
    parser.add_argument("--subject_name", type=str, required=True, help="Field name for the subject's answer (prediction).")
    args = parser.parse_args()
    
    evaluator = Evaluator(
        input_file=args.input_file,
        output_file=args.output_file,
        expert_name=args.expert_name,
        subject_name=args.subject_name
    )
    
    final_result = evaluator.evaluate()
    print("Final BLEURT result:")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
