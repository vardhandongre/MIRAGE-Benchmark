import sys
sys.path.append('../')
from chat_models.OpenAI_Chat import GPT4O
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

class ImageDescriptionExtractor:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):
        # Construct the prompt for image description
        prefix = "Now I need you to help me describe these agriculture-related images. I will provide you with a Q&A pair about these images, where a user asks a question and an expert responds. You can use these Q&A pairs to describe the images, but do not mention the presence of the expert. You can use the content of the expert's answer to describe these images."
        question_and_answer = f"Title: {item['title']}\n User: {item['question']}\n Expert: {item['answer']}"
        suffix = "Please describe these pictures in detail and high quality, in English: Please describe these pictures."
        prompt = prefix + question_and_answer + suffix
        return {"prompt": prompt, "images": item['attachments']}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        # Initialize GPT4O client
        client = GPT4O(model_name=model_name, messages=[])

        try:
            # Get the response from the model
            response = client.chat(prompt=prompt["prompt"], images=prompt["images"])
            item["description"] = response
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["description"] = None

        # Write the result to the output file
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        return item.get('id')

    def extract(self):
        # Read raw data
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Track already processed items
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "description" in item and item["description"] is not None:
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
                    if "description" in item and item["description"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image descriptions using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    extractor = ImageDescriptionExtractor(args.input_file, args.output_file, args.model_name, args.num_processes)
    extractor.extract()
