import sys
sys.path.append('../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
from typing import List


# Define a new Pydantic model for the direct answer check
class TAGS(BaseModel):
    image_description: str
    management_instructions: str
    miscellaneous_facts: List[str]
    
    def to_json(self):
        return {"image_description": self.image_description, "management_instructions": self.management_instructions, "miscellaneous_facts": self.miscellaneous_facts}

class Fact_Extractor:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):        
        prefix = """You are an assistant whose job it is to extract categories of information from an agriculture question. \
These categories are "image description", "management instructions", and "miscellaneous facts". Make sure that "image \
description" describe the visual symptoms that can be referenced in the hypothetical image. Make sure miscellaneous facts \
are independent of the exact situation and are standalone facts that don’t depend on location or the time of year. \
Format the response as a json. If no information for specific category is present, output "none".\n\n"""

        example = """<Example1>
CONTENT:
Invasive grass flower
Question Asker:
This small white flower invades the grass and multiplies and takes over the grass. It flowered in May but now in \
June stopped flowering. It roots are like a sweet potatoes and difficult to remove, you need to dig it up making \
your lawn look patchy.
Expert:
The weed you are trying to rid your lawn of is pennywort (Hydrocotyle americana, sometimes also known as \
dollarweed. It spreads by seed and by underground rhizomes and is a perennial that blooms early. It thrives in \
moist areas. The best way to control this broadleaf weed is to maintain a healthy lawn by regularly mowing at the \
recommended height for your variety of grass and watering deeply and infrequently to encourage deeper root \
growth. Monitor your lawn for areas that may need improved soil drainage. Fertilize your lawn appropriately; \
as recommended for your type of grass. Remove the weeds you presently have by hand pulling making sure to remove \
all roots. If your infestation is too broad to control by cultural methods, chemical control options are \
available. Use a herbicide designed to target this specific type of weed. Your local nursery operator can help \
you select the most effective application. When using any chemical read the label thoroughly and follow the \
instructions provided regarding the proper use and disposal Thank you for your question.

Model Response: {'image_description': 'small white flower with roots like sweet potato.', 'management_instructions': 'Maintain a \
healthy lawn by regularly mowing at the recommended height for your variety of grass and watering deeply and \
infrequently to encourage deeper root growth. Monitor your lawn for areas that may need improved soil drainage. \
Fertilize your lawn appropriately; as recommended for your type of grass. Remove the weeds you presently have by \
hand pulling making sure to remove all roots. If your infestation is too broad to control by cultural methods, \
chemical control options are available. Use a herbicide designed to target this specific type of weed', \
'miscellaneous_facts': ['pennywort spreads by seed and by underground rhizomes', 'pennywort is a perennial that \
blooms early', 'pennywort thrives in moist areas.']}
</Example1>


<Example2>
CONTENT:
Is this horsenettle?
Question Asker:
I am thinking that the attached photo is of horsenettle. Is that correct? I would prefer not to use any \
chemicals, but are there other ways to remove it premanently? It has such a long tap root when I try to dif it \
up.
Expert:
It does look like Carolina Horsenettle (Solanum carolinense), though flowers (or if it stuck around long enough, \
fruits) would help to confirm the ID. It is native, but considered a weed in garden and agricultural settings. \ 
Either systemic herbicide to kill the roots or vigilant physical removal would be needed to eradicate it. If you \
wish to avoid herbicide, then dig up (or cut down) what you can, and remove any regrowth as quickly as it \
appears. Eventually, this will starve the roots of stored energy, and the plant(s) will stop regrowing. How long \
this process takes is hard to predict, but it might be several months at least if the plant(s) is well- \
established or mature. Even herbicide might take more than one application to be successful. Miri

Model Response: {'image_description': 'has long taproot', 'management_instructions': 'Either systemic herbicide to kill the roots \
or vigilant physical removal would be needed to eradicate it. If you wish to avoid herbicide, then dig up (or cut \
down) what you can, and remove any regrowth as quickly as it appears. Eventually, this will starve the roots of \
stored energy, and the plant(s) will stop regrowing. ', 'miscellaneous_facts': ['Carolina Horsenettle is native \
but considered a weed in garden and agricultural settings']}
</Example2>
"""

        prompt = prefix + "Here are some examples:\n" + example + "\n\n" + f"CONTENT:\n{item['title']}\nQuestion Asker:\n{item['question']}\nExpert:\n{item['answer']}\n\nModel Response: "
        
        
        return {"prompt": prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            response = client.chat(prompt=prompt["prompt"], response_format=TAGS)
            item["tags"] = {"image_description": response.image_description, "management_instructions": response.management_instructions, "miscellaneous_facts": response.miscellaneous_facts}
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["tags"] = None
            
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
                        if "tags" in item and item["tags"] is not None:
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
                    if "tags" in item and item["tags"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facts from agriculture questions using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    extractor = Fact_Extractor(args.input_file, args.output_file, args.model_name, args.num_processes)
    extractor.extract()
