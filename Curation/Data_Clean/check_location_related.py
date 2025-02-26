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
class Judge(BaseModel):
    location_related: bool
    
    def to_json(self):
        return {"location_related": self.location_related}

class CheckLocationRelated:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is satisfactory
    def get_prompt(self, item):
        prompt = """I am processing an agricultural Q&A dataset and need assistance in identifying whether each Q&A pair is location contextual. The first scenario involves a question that does not mention a specific location, yet the answer suddenly includes one. This database is sourced from a website where experts who answer questions can see the location of the user asking the question, even if the user does not mention this location in their question. However, be aware that if the location information is a matter of knowledge, such as when an expert identifies an insect and states that it is commonly found in a certain area, this data should be retained. The second scenario involves the user or expert mentioning vague locations, such as "my town" or "my area," which are not specific. This can result in an imbalance of information between the question and answer. However, more general terms like "my yard," "my gardens," or "my houses" do not qualify as vague locations.

Please read the following examples carefully and use them as a basis for your judgment.

<Example 1: Expert mentioned Montgomery County' Law, but the user did not mention any location>
User Question: We recently got some landscaping done at our house and now notice we have North American Oil Beetles in our yard. We have noticed three clumps of them and several scattered in the grass. Our internet research says they can cause painful blisters and I fear we have too many to remove by hand. What should and can be done to remove them? Thanks so much.
Expert Answer: These are not ordinarily a problem in yards. Since these are damaging to bee populations, it would be good to eradicate them or at least bring down the population to the usual insignificant levels. You cannot spray pesticides on your lawn in Montgomery County. We'd recommend that you not try to hand-pick them from the lawn, but simply walk on them and crush them as much as possible. Leave the bodies to decompose. If you have a significant population of them next year, please contact us again. These insects eat vegetative matter, particularly buttercups. Buttercups are fairly easy to pull when the soil is moist. You might want to remove buttercups if you have them around.

Model Output: 
{"location_related":True}
<Example 1 End>

<Example 2: User mentioned Larkspur in Douglas County>
User Query: Hello, I found this plant has aggressively grown on disturbed soil in the area around Larkspur in Douglas County. I have not found it on the noxious weeds list by the County or State and was wondering what the identity is and if I should be getting rid of it.
Expert Response: It may be a white heath aster; it is hard to tell without the flowers being opened. It is not a noxious weed. One person's flower is another person's weed, so it is up to your needs as to whether you should get rid of it. I believe it will continue to spread as it produces many seeds and will not go away on its own.

Model Output:
{"location_related":False}
<Example 2 End>

<Example 3: Expert mentioned "our area", which is not clear>
User Query: Trying to figure out if this will damage my lilacs. It started on this one and has moved to others nearby. What can I do? My lilacs are 15-19 years old.
Expert Response: The lichen appearing on your lilacs suggests humidity and slow growth. An older lilac shrub growing slowly is natural and not necessarily a reason for alarm. The lichen is not corrupting the plant on its own. But while the lichen is not causing a problem, it may be a signal that your lilacs are in distress. We have seen a lot of lichen appearing on trees in our area this summer as it was unusually moist in the early months. It also is possible that your tree is in too protected a position where moisture is accumulating and that any thinning of nearby plants that you can do to increase air flow will have a benefit.
Model Output:
{"location_related":True}
<Example 3 End>

<Example 4: my yard is not considered an unclear location>
User Query: I get these small mounds of dirt in my yard. I am new to the area and haven't come across this before. I need to identify them so I know how to treat them. It wouldn't be so bad if there weren't so many.
Expert Response: Sorry for the slow reply. This is likely from earthworms. This event often happens after a good soaking rain. It is nothing to worry about, they are aerating the soil for you. The mounds can be raked down if you don't like the way they look. Hope this is helpful.

Model Output:
{"location_related":False}
<Example 4 End>

<Example 5: Expert mentioned "southern part of the lower peninsula", but this one is a knowledge-based question, even user did not mention any location, expert can provide the answer based on the knowledge>
User Query: I spotted this yesterday in my backyard, visiting the boneset flowers. Is this a grizzled skipper? From what I can find, that species is not typically found in my county. It had about a 4cm wingspan. Thank you!
Expert Response: That is a checkered skipper, Pyrgus communis. It is a common skipper in the southern part of the lower peninsula.

Model Output:
{"location_related":False}
<Example 5 End>

Please judge the following Q&A pairs. 
"""

        sample_prompt = f"User Query: {item['question']}\nExpert Response: {item['answer']}\n\nModel Output:\n"
        return {"prompt": prompt + sample_prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            response = client.chat(prompt=prompt["prompt"], response_format=Judge)
            item["location_related"] = response.location_related
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["location_related"] = None
            
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
                        if "location_related" in item and item["location_related"] is not None:
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
                    if "location_related" in item and item["location_related"] is not None:
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

    checker = CheckLocationRelated(args.input_file, args.output_file, args.model_name, args.num_processes)
    checker.extract()
