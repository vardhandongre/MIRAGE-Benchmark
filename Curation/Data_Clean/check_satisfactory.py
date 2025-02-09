import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

# Define a new Pydantic model for the direct answer check
class Satisfactory(BaseModel):
    satisfactory: bool
    
    def to_json(self):
        return {"satisfactory": self.satisfactory}

class CheckSatisfactory:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is satisfactory
    def get_prompt(self, item):        
        prefix = """\
I am cleaning an agricultural Q&A dataset and need your help to determine \
if the expert's answer is satisfactory. If the expert's answer is unsatisfactory, \
such as suggesting to contact someone else, asking for more information, \
indicating uncertainty, or admitting they cannot help, output {"satisfactory": False}. Otherwise, output {"satisfactory": True}. 
Here are some examples to help you understand the task better:

<Example1: suggesting to contact someone else>
Expert's Answer:
Good morning Dave,  Thank you for contacting AnswerLine. I would recommend contacting the Linn County Master Gardeners for assistance with your question. They can be reached at <personal data hidden> or their Hortline at<personal data hidden> (Monday - Thursday from 10AM-12PM).'
Judgement: 
{"satisfactory": False}
<Example1>

<Example2: asking for more information>
Expert's Answer: 
Dahlias can decline due to fungal diseases, but there are also bacterial and virus problems. Knowing which it is for sure is tough, and sometimes there are multiple issues. Review best cultural practices as listed in FS 95. Are you following those? How did these dahlias grow last year? Are you digging and replanting yearly? Pull up one of the small sick ones. How are the roots? Do they look and smell rotten? Do you find slugs and snails eating the tubers below ground?
Judgement: 
{"satisfactory": False}
<Example2>

<Example3: indicating uncertainty>
Expert's Answer:
We can not tell for sure from your photos what your issue is. Possibilities include wildlife burrowing and stormwater issues.
Judgement: 
{"satisfactory": False}
<Example3>

<Example4: admitting they cannot help>
Expert's Answer:
Sorry, but our Cooperative Extension System experts can't identify plants outside the U.S.
Judgement: 
{"satisfactory": False}
<Example4>

<Example5: Satisfactory answer>
Expert's Answer:
Greetings, This fungus is Phaeolus schweinitzii, the "dyers polypore," so-called because it is used as a dyestuff by crafters.  It is not edible.
Judgement:
{"satisfactory": True}
<Example5>

Please judge the following answer. 
"""
        sample_prompt = f"Expert's Answer:\n{item['answer']}\nJudgement:\n"
        return {"prompt": prefix + "\n" + sample_prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            response = client.chat(prompt=prompt["prompt"], response_format=Satisfactory)
            item["satisfactory"] = response.satisfactory
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["satisfactory"] = None
            
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def extract(self):
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "satisfactory" in item and item["satisfactory"] is not None:
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
                    if "satisfactory" in item and item["satisfactory"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if expert answers are satisfactory.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    checker = CheckSatisfactory(args.input_file, args.output_file, args.model_name, args.num_processes)
    checker.extract()
