import sys
sys.path.append('../')
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
    relevance: int
    completeness: int

    def to_json(self):
        return {
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "completeness": self.completeness
        }

class Scorer:
    def __init__(self, raw_data_file, output_file, expert_name, subject_name, model_name="gpt-4o", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.expert_name = expert_name
        self.subject_name = subject_name

    def get_prompt(self, item):
        title = item["title"]
        user_query = item["question"]
        if self.expert_name not in item:
            if self.expert_name in item["b_type"]:
                expert_answer = item["b_type"][self.expert_name]
            else:
                raise ValueError(f"Expert answer not found in item {item.get('id', 'unknown')}")
        else:
            expert_answer = item[self.expert_name]
        images = item.get("attachments", [])
        if self.subject_name not in item:
            if self.subject_name in item["a_type"]:
                model_response = item["a_type"][self.subject_name]
            elif self.subject_name in item["b_type"]:
                model_response = item["b_type"][self.subject_name]
            else:
                raise ValueError(f"Model response not found in item {item.get('id', 'unknown')}")
        else:
            model_response = item[self.subject_name]

        score_criteria = """\
**1. Accuracy (1-5 points)**

- **5 points:** The response perfectly reflects all key information and details in the image, such as the correct crop, pests, etc., and is completely consistent with the perfect answer in terms of accuracy.
- **4 points:** The response correctly reflects most of the key information and details in the image, with only minor errors.
- **3 points:** The response is partially correct but omits some important information or contains some errors.
- **2 points:** The response has many errors and only a small portion of the information is accurate.
- **1 point:** The response almost entirely does not match the key information and details in the image, with numerous errors.

**2. Relevance (1-5 points)**

- **5 points:** The response is direct and fully addresses the question, effectively solving the problem.
- **4 points:** The response mainly addresses the question, largely solving the problem but includes some minor irrelevant content.
- **3 points:** The response is related to the question but does not fully focus on the core of the question, or contains a significant amount of irrelevant information.
- **2 points:** The response has little relevance to the question, only touching on certain aspects of it.
- **1 point:** The response is almost entirely unrelated to the question, or does not solve the problem at all.

**3. Completeness (1-5 points)**

- **5 points:** The response comprehensively covers all the key information needed for the question, with no omissions.
- **4 points:** The response covers most of the key information, but misses some more minor details.
- **3 points:** The response covers some of the key information but misses quite a lot.
- **2 points:** The response omits a large amount of key information, providing only limited information.
- **1 point:** The response almost does not cover any key information, with extensive omissions.
"""
        prompt = f"""
You are now required to rate a model's response to an agriculture-related question. \
We have a perfect answer, which is Expert's Answer and based on this perfect answer, the user-provided image, \
and the user's question, you need to score the model's answer according to the following three scoring criteria.

<Title>{title}</Title>

<User Query>{user_query}</User Query>

<Expert Answer>{expert_answer}</Expert Answer>

<Model Response>{model_response}</Model Response>

<Score Criteria>{score_criteria}</Score Criteria>

Please only output the scores without any other content. You should output JSON with three key,accuracy,relevance,completeness. The example is shown below:
{{ "accuracy": ..., "relevance": ..., "completeness": ... }}"""

        system_prompt = "You are a helpful assistant that evaluates and scores responses in agricultural contexts."
        
        return {
            "prompt": prompt,
            "system": system_prompt,
            "images": images,
            "expert_answer": expert_answer,
            "model_response": model_response
        }

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)
        max_retries = 5
        retries = 0
        while retries < max_retries:
            if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
                client = GPT4O(model_name=model_name, messages=[])
            elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash":
                client = Gemini(model_name=model_name, messages=[])
            elif self.model_name == "claude-3-5-sonnet-latest":
                client = Claude(model_name=model_name, messages=[])
            else:
                raise ValueError(f"Model '{self.model_name}' not supported.")
            
            new_item = {
                "jugde": self.model_name,
                "id": item.get('id'),
                "title": item.get('title'),
                "question": item.get('question'),
                "attachments": item.get('attachments'),
                "expert_name": self.expert_name,
                "subject_name": self.subject_name,
                "expert_answer": prompt["expert_answer"],
                "model_response": prompt["model_response"]
            }
            
            try:
                if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
                    response = client.chat(prompt=prompt["prompt"], images=prompt["images"], response_format=Score)
                    new_item["score"] = response.to_json()
                    response = response.to_json()
                elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash":
                    response = client.chat(prompt=prompt["prompt"], images=prompt["images"])
                    response = self.extract_json(response)
                    new_item["score"] = response
                elif self.model_name == "claude-3-5-sonnet-latest":
                    client.system = prompt["system"] + "\nOutput only valid JSON with exactly this format: {\"accuracy\": score, \"relevance\": score, \"completeness\": score}"
                    response = client.chat(prompt=prompt["prompt"], images=prompt["images"])
                    response = self.extract_json(response)
                    new_item["score"] = response
                
                assert "accuracy" in response and "relevance" in response and "completeness" in response
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
                        if 'score' in item and item['score'] != -1 and item['score'] != None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        # Handle potentially corrupt JSON lines
                        continue
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            # Initialize the process pool with the specified number of processes
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock) for item in items_to_process]
            # Use tqdm to show progress
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
                    if 'score' in item and item['score'] != -1 and item['score'] != None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--expert_name", type=str, required=True, help="Name of the expert's answer field.")
    parser.add_argument("--subject_name", type=str, required=True, help="Name of the subject's answer field.")
    
    args = parser.parse_args()
    scorer = Scorer(args.input_file, args.output_file, args.expert_name, args.subject_name, args.model_name, args.num_processes)
    scorer.scoring()
