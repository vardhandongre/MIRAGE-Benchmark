import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from chat_models.Gemini import Gemini
from chat_models.Claude import Claude  # Add Claude import
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
import re

class Score(BaseModel):
    accuracy: int

    def to_json(self):
        return {"accuracy": self.accuracy}

class Scorer:
    def __init__(self, raw_data_file, output_file, expert_name, subject_name, model_name="gpt-4o", num_processes=None, temperature=1):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.expert_name = expert_name
        self.subject_name = subject_name
        self.temperature = temperature

    def get_prompt(self, item):
        user_query = item["question"]
        # Retrieve expert answer
        if self.expert_name not in item:
            if self.expert_name in item.get("b_type", {}):
                expert_answer = item["b_type"][self.expert_name]
            else:
                raise ValueError(f"Expert answer not found in item {item.get('id', 'unknown')}")
        else:
            expert_answer = item[self.expert_name]
        # Retrieve model response
        if self.subject_name not in item:
            if self.subject_name in item.get("a_type", {}):
                model_response = item["a_type"][self.subject_name]
            elif self.subject_name in item.get("b_type", {}):
                model_response = item["b_type"][self.subject_name]
            else:
                raise ValueError(f"Model response not found in item {item.get('id', 'unknown')}")
        else:
            model_response = item[self.subject_name]

        category = item.get("category", "unknown")
        assert category in ["Plant Disease Identification", "Plant Identification", "Insect and Pest Identification"]
        
        if category == "Plant Disease Identification":
            entity = "plant disease"
        elif category == "Plant Identification":
            entity = "plant"
        elif category == "Insect and Pest Identification":
            entity = "insect/pest"
        
        prompt = f"""
You are tasked with evaluating a {entity} identification task.
Below are the User's question, Expert's answer and the Model's answer.

<Question>: {user_query}

<Expert Answer>: {expert_answer}

<Model Answer>: {model_response}

Determine whether the Model's answer identifies the same {entity} as the Expert's answer.
If they refer to the same {entity}, output exactly:
{{"accuracy": 1}}

If they are different, output exactly:
{{"accuracy": 0}}

Please output only the JSON object without any additional text.
"""
        
        return {
            "prompt": prompt,
            "expert_answer": expert_answer,
            "model_response": model_response
        }

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)
        max_retries = 5
        retries = 0
        while retries < max_retries:
            if self.model_name.startswith("gpt"):
                client = GPT4O(model_name=model_name, messages=[])
            elif self.model_name in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]:
                client = Gemini(model_name=model_name, messages=[])
            elif self.model_name in ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"]:
                client = Claude(model_name=model_name, messages=[])
            else:
                raise ValueError(f"Model '{self.model_name}' not supported.")
            
            new_item = {
                "judge": self.model_name,
                "id": item.get('id'),
                "question": item.get('question'),
                "attachments": item.get('attachments'),
                "expert_name": self.expert_name,
                "subject_name": self.subject_name,
                "expert_answer": prompt["expert_answer"],
                "model_response": prompt["model_response"],
                "category": item.get("category"),
            }
            
            try:
                if self.model_name.startswith("gpt"):
                    response = client.chat(prompt=prompt["prompt"], response_format=Score, temperature=0)
                    new_item["score"] = response.to_json()
                    response = response.to_json()
                elif self.model_name in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]:
                    response = client.chat(prompt=prompt["prompt"], response_format=Score, temperature=0)
                    response = self.extract_json(response)
                    new_item["score"] = response
                elif self.model_name in ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"]:
                    response = client.chat(prompt=prompt["prompt"])
                    response = self.extract_json(response)
                    new_item["score"] = response
                
                # Check only the accuracy key is present.
                assert "accuracy" in response
                new_item["info"] = client.info()
                new_item["history"] = client.get_history()
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error processing item {item.get('id', 'unknown')}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                new_item["score"] = -1
                
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        
        return new_item.get('id')

    def extract_json(self, string):
        try:
            json_data = json.loads(string)
            return json_data
        except json.JSONDecodeError:
            if isinstance(string, dict):
                return string
            
            pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
            match = re.search(pattern, string)

            if match:
                json_str = match.group(1)
            else:
                # Look for JSON object in the string
                start = string.find('{')
                end = string.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = string[start:end]
                else:
                    json_str = string.strip()

            json_data = json.loads(json_str)   
            return json_data   

    def scoring(self):
        # Read the raw data file
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Check if the output file exists and read processed items
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'score' in item and item['score'] != -1 and item['score'] is not None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        continue
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock) for item in items_to_process]
            for _ in tqdm(pool.imap_unordered(self.process_item, args_list), total=len(args_list), desc="Processing items"):
                pass
            pool.close()
            pool.join()
        
        print("Processing completed.")
        self.cleanup_output(len(data))

    def cleanup_output(self, data_length):
        valid_items = []
        
        with open(self.output_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'score' in item and item['score'] != -1 and item['score'] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \nRemaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score plant identification responses using LLMs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--expert_name", type=str, required=True, help="Name of the expert's answer field.")
    parser.add_argument("--subject_name", type=str, required=True, help="Name of the subject's answer field.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model response generation.")
    
    args = parser.parse_args()
    scorer = Scorer(args.input_file, args.output_file, args.expert_name, args.subject_name, args.model_name, args.num_processes, args.temperature)
    scorer.scoring()
