from OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

class Generate:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        # If the number of processes is not specified, use the number of CPU cores
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):
        
        title = item["title"]
        question = item["question"]
        images = item.get("attachments", [])
        user_prompt = f"Title: {title}\nQuestion: {question}"
        
        return {"user": user_prompt, "images": images}

    # Function to handle item processing
    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)
        client = GPT4O(model_name=model_name, messages=[])
        
        try:
            response = client.chat(prompt=prompt["user"], images=prompt["images"])
            item[model_name] = response
            
        except Exception as e:
            # Handle errors gracefully and log them
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item[model_name] = -1
            
        # Lock the file access to avoid race conditions
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                # Each item is written as a single JSON line
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def generate(self):
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
                        if self.model_name in item and item[self.model_name] != -1:
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
                    if self.model_name in item and item[self.model_name] != -1:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    reformatter = Generate(raw_data_file=args.input_file, output_file=args.output_file, model_name=args.model_name, num_processes=args.num_processes)
    reformatter.generate()
