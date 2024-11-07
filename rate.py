from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

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
        # If the number of processes is not specified, use the number of CPU cores
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        
        self.expert_name = expert_name
        self.subject_name = subject_name
        

    def get_prompt(self, item):
        title = item["title"]
        user_query = item["question"]
        expert_answer = item[self.expert_name]
        images = item.get("attachments", [])
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
We have a perfect answer, and based on this perfect answer, the user-provided image, \
and the user's question, you need to score the model's answer according to the following three scoring criteria.

<Title>{title}</Title>

<User Query>{user_query}</User Query>

<Expert Answer>{expert_answer}</Expert Answer>

<Model Response>{model_response}</Model Response>

<Score Criteria>{score_criteria}</Score Criteria>

Please only output the scores without any other content."""  
        
        return {"prompt": prompt, "images": images}

    # Function to handle item processing
    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)
        client = GPT4O(model_name=model_name, messages=[])
        
        new_item = {
            "id": item.get('id'),
            "title": item.get('title'),
            "question": item.get('question'),
            "attachments": item.get('attachments'),
            "expert_name": self.expert_name,
            "subject_name": self.subject_name,
            "expert_answer": item.get(self.expert_name),
            "model_response": item.get(self.subject_name)
        }
        
        try:
            response = client.chat(prompt=prompt["prompt"], images=prompt["images"], response_format=Score)
            new_item["score"] = response.to_json()
        
        except Exception as e:
            # Handle errors gracefully and log them
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            new_item["score"] = -1
            
        # Lock the file access to avoid race conditions
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                # Each item is written as a single JSON line
                f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        
        return new_item.get('id')

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
                        if 'score' in item and item['score'] != -1:
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
                    if 'score' in item and item['score'] != -1:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--expert_name", type=str, required=True, help="Name of the expert's answer field.")
    parser.add_argument("--subject_name", type=str, required=True, help="Name of the subject's answer field.")
    
    args = parser.parse_args()
    scorer = Scorer(args.input_file, args.output_file, args.expert_name, args.subject_name, args.model_name, args.num_processes)
    scorer.scoring()
