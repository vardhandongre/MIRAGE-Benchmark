import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

class Entity(BaseModel):
    entity: str

    def to_json(self):
        return {"entity": self.entity}

class ExtractEntity:
    def __init__(self, input_file, output_file, model_name="gpt-4o-mini", num_processes=None, plant_only=False):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        # Extract Plant Only
        self.plant_only = plant_only
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):
        if self.plant_only:
                prompt = """Extract the main plants mentioned in the Q&A pairs. Please try to extract specific plant names, keeping them as short as possible (either one word or two words, all in lowercase and in the singular form). If no plant name is mentioned, output "none". Present the result in JSON format as follows: {"entity": ...}\n\n"""
                prompt += f"Q: {item['question']}\nA: {item['answer']}"
        if not self.plant_only:
            if item.get("category") in [
                "Plant Identification", 
                "Plant Care and Gardening Guidance"
            ]:
                prompt = """Extract the main plants mentioned in the Q&A pairs. Please try to extract specific plant names, keeping them as short as possible (either one word or two words, all in lowercase and in the singular form). If no plant name is mentioned, output "none". Present the result in JSON format as follows: {"entity": ...}\n\n"""
                prompt += f"Q: {item['question']}\nA: {item['answer']}"
            elif item.get("category") in [
                "Weeds/Invasive Plants Management"
            ]:
                prompt = """Extract the main weed/invasive plant mentioned in the Q&A pairs. Please try to extract one specific weed/invasive plant name, keeping it as short as possible (either one word or two words, all in lowercase and in the singular form). If no weed/invasive plant name is mentioned, output "none". Present the result in JSON format as follows:{"entity": ...}\n\n"""
                prompt += f"Q: {item['question']}\nA: {item['answer']}"
                
            elif item.get("category") in [
                "Insect and Pest Identification", 
                "Insect and Pest Management"
            ]:
                prompt = """Extract the main insect and pest mentioned in the question-answer pair.  Please try to extract specific insect or pest names, keeping them as short as possible (either one word or two words, all in lowercase and in the singular form). If no insect or pest is mentioned, output "none". Present the result in JSON format as follows: {"entity": ...}\n\n"""
                prompt += f"Q: {item['question']}\nA: {item['answer']}"
            elif item.get("category") in [
                "Plant Disease Identification",
                "Plant Disease Management"
            ]:
                prompt = """Extract the main plant disease mentioned in the question-answer pair. Please try to extract specific plant disease, keeping them as short as possible (either one word or two words, all in lowercase and in the singular form). If no plant disease is mentioned, output "none".  Present the result in JSON format as follows: {"entity": ...}\n\n"""
                prompt += f"Q: {item['question']}\nA: {item['answer']}"
            else:
                print(f"Category '{item.get('category')}' not supported.")
                prompt = (
                    'Extract the main entity (plant, insect, or disease) mentioned in the question-answer pair. '
                    'If none is mentioned, output "None". '
                    'Present the result in JSON format as follows: {"entity": ...}\n\n'
                    f'Q: {item["question"]}\nA: {item["answer"]}'
                )
        
        return {"prompt": prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt_data = self.get_prompt(item)
        
        if model_name in ["gpt-4o", "gpt-4o-mini"]:
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{model_name}' not supported.")
    
        try:
            retries = 3  # Maximum number of retries

            while retries > 0:
                if self.plant_only:
                    if item.get("category") in [
                                    "Plant Identification", 
                                    "Plant Care and Gardening Guidance"
                                ]:
                        item["plant"] = item["entity"]
                        break  # No need to retry if category matches
                    else:
                        response = client.chat(prompt=prompt_data["prompt"], response_format=Entity)
                        if response.entity != "none":
                            item["plant"] = response.entity
                            item["info"] = client.info()
                            break
                else:
                    response = client.chat(prompt=prompt_data["prompt"], response_format=Entity)
                    if response.entity != "none":
                        item["entity"] = response.entity
                        item["info"] = client.info()
                        break
                
                retries -= 1  # Decrement the retries counter
                if retries > 0:
                    print(f"Retrying for item {item.get('id', 'unknown')}... {3 - retries} retries left.")
                    client = GPT4O(model_name=model_name, messages=[])  # Re-initialize client for a fresh call

            if retries == 0 and response.entity == "none":
                print(f"Failed to process item {item.get('id', 'unknown')}: entity could not be retrieved after 3 attempts.")
                if self.plant_only:
                    item["plant"] = "none"
                else:
                    item["entity"] = "none"

        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            if self.plant_only:
                item["plant"] = None
            else:
                item["entity"] = None

        item["retry_count"] = 3 - retries
        
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
        return item.get('id')


    def extract(self):
        with open(self.input_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if not self.plant_only and "entity" in item and item["entity"] is not None:
                            processed_ids.add(item['id'])
                        elif self.plant_only and "plant" in item and item["plant"] is not None:
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
            processed_ids = set()
            for line in f:
                try:
                    item = json.loads(line)
                    if not self.plant_only and "entity" in item and item["entity"] is not None:
                        processed_ids.add(item['id'])
                        valid_items.append(item)

                    elif self.plant_only and "plant" in item and item["plant"] is not None:
                        processed_ids.add(item['id'])
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}.")
        print(f"Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entity (plant, insect/pest, or disease) based on category using LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--plant_only", action="store_true", help="Extract plant entity only.")
    args = parser.parse_args()

    extractor = ExtractEntity(input_file=args.input_file, output_file=args.output_file, model_name=args.model_name, num_processes=args.num_processes, plant_only=args.plant_only)
    extractor.extract()