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
    knowledge_location_related: bool
    
    def to_json(self):
        return {"knowledge_location_related": self.knowledge_location_related}

class CheckLocationRelated:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is satisfactory
    def get_prompt(self, item):
        prompt = """I'm processing an agricultural Q&A dataset and need assistance in identifying whether in each Q&A pair an expert claims that a particular plant or animal often appears in a specific area. If so, output True. However, if the expert mentions a vague place like "my area," output False.

Please carefully read the examples below and judge accordingly.

<Example 1: Expert mentioned Montgomery County' Law, but not belong to our case>
User Question: We recently got some landscaping done at our house and now notice we have North American Oil Beetles in our yard. We have noticed three clumps of them and several scattered in the grass. Our internet research says they can cause painful blisters and I fear we have too many to remove by hand. What should and can be done to remove them? Thanks so much.
Expert Answer: These are not ordinarily a problem in yards. Since these are damaging to bee populations, it would be good to eradicate them or at least bring down the population to the usual insignificant levels. You cannot spray pesticides on your lawn in Montgomery County. We'd recommend that you not try to hand-pick them from the lawn, but simply walk on them and crush them as much as possible. Leave the bodies to decompose. If you have a significant population of them next year, please contact us again. These insects eat vegetative matter, particularly buttercups. Buttercups are fairly easy to pull when the soil is moist. You might want to remove buttercups if you have them around.

Model Output: 
{"knowledge_location_related":False}
<Example 1 End>

<Example 2: Expert mentioned "our area", which is not clear>
User Query: Trying to figure out if this will damage my lilacs. It started on this one and has moved to others nearby. What can I do? My lilacs are 15-19 years old.
Expert Response: The lichen appearing on your lilacs suggests humidity and slow growth. An older lilac shrub growing slowly is natural and not necessarily a reason for alarm. The lichen is not corrupting the plant on its own. But while the lichen is not causing a problem, it may be a signal that your lilacs are in distress. We have seen a lot of lichen appearing on trees in our area this summer as it was unusually moist in the early months. It also is possible that your tree is in too protected a position where moisture is accumulating and that any thinning of nearby plants that you can do to increase air flow will have a benefit.
Model Output:
{"knowledge_location_related":False}
<Example 2 End>


<Example 3: Expert mentioned It is a common skipper in the southern part of the lower peninsula, which is a specific location, and contains a plant or animal name>
User Query: I spotted this yesterday in my backyard, visiting the boneset flowers. Is this a grizzled skipper? From what I can find, that species is not typically found in my county. It had about a 4cm wingspan. Thank you!
Expert Response: That is a checkered skipper, Pyrgus communis. It is a common skipper in the southern part of the lower peninsula.

Model Output:
{"knowledge_location_related":True}
<Example 3 End>

<Example 4: Expert mentioned unclear location "There is a soil testing lab nearby and the local extension office">
User Query: Several years ago we replaced a railroad tie retaining wall with a rock wall. In the process we removed all the existing shrubs and trees along with some soil. When the new wall was complete we filled in with top soil and then a couple inches of hemlock bark. This has been in place three years. We are now planning to plant shrubs and a couple small trees in the area. Should we do a general augmentation to all of the soil or add the appropriate nutrients to each specific plant? Attached are three photos of the site.
Expert Response: Very nice look to the front. Would you consider doing a soil test to that soil to know just what is needed and what is not? It is not too expensive and may save money and effort later on. It will tell you the pH and the nutrients already present. Has the bark all broken down by now? Shrubs and new trees will require much the same nutrition as a rule. There is a soil testing lab nearby and the local extension office has a listing of soil labs. If you still have unanswered concerns, please just get right back to me.

Model Output:
{"knowledge_location_related":False}
<Example 4 End>

<Example 5: Expert mentioned "native to the region", which is not clear>
User Query: I have tried to ID this plant all summer and the Seek app only gives me dicots. It has tiny purple flowers. I don't remember if I even planted it and want to find out if it is native or not. Seems to have many seed pods now and I want to know more so I can take the best course of action before the seeds spread. Thank you so much!!!
Expert Response: This looks like Lobelia inflata (Indian tobacco) which is native to the region. It looks like an interesting plant to keep around but just be aware that it can be toxic to humans if eaten in large amounts. Its common name is pukeweed. I hope this helps!

Model Output:
{"knowledge_location_related":False}
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
            item["knowledge_location_related"] = response.knowledge_location_related
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["knowledge_location_related"] = None
            
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
                        if "knowledge_location_related" in item and item["knowledge_location_related"] is not None:
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
                    if "knowledge_location_related" in item and item["knowledge_location_related"] is not None:
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
