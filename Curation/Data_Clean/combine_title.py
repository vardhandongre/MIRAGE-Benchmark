import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

class TA(BaseModel):
    question: str
    def to_json(self):
        return {"question": self.question}

class ReformatQ:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):        
        prompt = """\
We are processing an agricultural Q&A dataset scraped from a website where users post a title and a question, and experts provide answers. Your task is to review the given title and question, and if necessary, merge the information from the title into the question to create a new, unified question. It is crucial to preserve all information and modify the original question as minimally as possible.

Requirements:
- Examine the provided title and question.
- Integrate any additional information from the title into the question without losing any details.
- Maintain the original content of the question as much as possible.
- If modifications are needed, output a JSON object in the following format: {"question": "<modified question>"}
- If no modifications are necessary, output: {"question": False}

Please read the following examples carefully and use them as a basis for your output:

<Example 1>
Title: What insect is this?
User Query: It is smaller than a thumbtack head. I found it in my classroom and it was already dead.

Model Output:
{"question": "What insect is this? It is smaller than a thumbtack head. I found it in my classroom and it was already dead."}
<Example 1 End>

<Example 2>
Title: what is wrong with my rhubarb plants?
User Query: Any help diagnosing this problem would be much appreciated? They have been in this location for years and normally are very robust. This problem has been getting worse over the last 3 years or so.

Model Output:
{"question": "What is wrong with my rhubarb plants? Any help diagnosing this problem would be much appreciated? They have been in this location for years and normally are very robust. This problem has been getting worse over the last 3 years or so."}
<Example 2 End>

<Example 3>
Title: Nootka cypress
User Query: Can you tell me why these trees are turning brown?

Model Output:
{"question": "Can you tell me why these Nootka cypress trees are turning brown?"}
<Example 3 End>

<Example 4>
Title: What is this insect?
User Query: I found this insect flying around the field. What is it?

Model Output:
{"question": False}
<Example 4 End>

<Example 5>
Title: What is this?
User Query: What is the plant found in a pot with other plants?

Model Output:
{"question": False}
<Example 5 End>

Please review the following title and question.
"""

        prompt += f"\n\nTitle: {item['title']}\nUser Query: {item['question']}\nModel Output:\n"

        return {"prompt": prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name.startswith("gpt"):
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            response = client.chat(prompt=prompt["prompt"], response_format=TA)
            item["new_question"] = response.question
            item["info"] = client.info()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["new_question"] = None
            
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def reformatting(self):
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "new_question" in item and item["new_question"] is not None:
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
                    if "new_question" in item and item["new_question"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat questions using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    reformat = ReformatQ(args.input_file, args.output_file, args.model_name, args.num_processes)
    reformat.reformatting()
