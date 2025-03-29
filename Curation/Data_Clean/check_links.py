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
class Link_Problem(BaseModel):
    links_problem: bool
    
    def to_json(self):
        return {"links_problem": self.links_problem}

class CheckLinkProblem:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is links_problem
    def get_prompt(self, item):
        prefix = """\
I am cleaning an agricultural Q&A dataset and need your help to determine if the expert's answer has a link problem. Analyze each expert answer and respond with ONLY {"links_problem": True} or {"links_problem": False}.

An expert's answer is considered to have a link problem (output {"links_problem": True}) if it:
1. Mentions that a link exists when no actual link is provided.
2. Includes actual links.

An expert's answer is considered free of link problems (output {"links_problem": False}) only if it does not mention or include any links or link-like instructions.

Examples:

<Example1>
Expert's Answer:
I believe that is early Black Knot. Please open the drop down boxes in this link to deal with it ASAP. Black knot | UMN Extension
Judgment:
{"links_problem": True}
Reason: The answer mentions a link and provides link text.

<Example2>
Expert's Answer:
Hi, The grass in your photograph seems to be Poa trivialis, or Roughstalk bluegrass, which you can read about by clicking on the link. You will see that the only method for control is hand-digging. The plant with blue flowers is wild violet, which you can read about by clicking on the link. Thank you for your question.
Judgment:
{"links_problem": True}
Reason: The answer repeatedly mentions links without providing any actual link details.

<Example3>
Expert's Answer:
Hello, happy to help. Apple scab is a common problem during cool, wet springs. If you compost, be sure your compost reaches temperatures that kill fungi and weed seeds. If not, sending your yard waste to a city processor is the better option. Here is more information about apple scab. Good luck!
Judgment:
{"links_problem": True}
Reason: The answer mentions additional information is available via a link, even though no link is actually provided.

<Example4>
Expert's Answer:
This fungus is Phaeolus schweinitzii, the "dyers polypore." It is not edible.
Judgment:
{"links_problem": False}
Reason: The answer does not mention or include any links.

Please judge the following answer:
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
            response = client.chat(prompt=prompt["prompt"], response_format=Link_Problem)
            item["links_problem"] = response.links_problem
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["links_problem"] = None
            
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
                        if "links_problem" in item and item["links_problem"] is not None:
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
                    if "links_problem" in item and item["links_problem"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if expert answers are links_problem.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    checker = CheckLinkProblem(args.input_file, args.output_file, args.model_name, args.num_processes)
    checker.extract()
