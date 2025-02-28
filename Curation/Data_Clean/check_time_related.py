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
    time_related: bool
    
    def to_json(self):
        return {"time_related": self.time_related}

class CheckTimeRelated:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is satisfactory
    def get_prompt(self, item):
        prompt = """I am processing an agricultural Q&A dataset. Our dataset was crawled from a website where experts can see the time of users' questions when answering them. However, this time is not included in the users' questions. I need to determine if this time was used in the Q&A, and not knowing this time would affect the content of the Q&A. Then output {"time_related":True}. \\
If the time was not used in the Q&A, or but not knowing the exact user's time would not affect the content of the Q&A, then output {"time_related":False}. \\

Please read the following examples carefully and use them as a basis for your judgment.

<Example 1: Expert mentioned "They only live a season and usually die around this time of year", used the time information>
User Query: I found this spider at a property. Can anyone help identify it?
Expert Response: Hello, this appears to be a variation of the orb weaver spider. They are a beneficial spider that clean up the garden of insects. They are usually hesitant to bite - although like all spiders, if instigated, they will bite. They only live a season and usually die around this time of year. Let me know if you have any further questions :)

Model Output: 
{"time_related":True}
<Example 1 End>

<Example 2: Expert mentioned "really the past two years have been very dry", used the time information>
User Query: My tree tops are dying. There are about four dead/dying trees in a bunch of about six. Is this from a drier than normal summer and the trees could not survive at their current density? I do not see any signs of insect infestation but may have missed something. Should I go ahead and cut these trees down and consider them dead?
Expert Response: Yes, I suspect the dry conditions of the past summer - really the past two years have been very dry - have resulted in the dieback and death of these trees. There may also be bugs or disease involved but the underlying reason is likely to be moisture stress. 

Model Output:
{"time_related":True}
<Example 2 End>

<Example 3: Expert mentioned "through the winter" and "in the spring", but not knowing the exact user's time  would not affect the content of the Q&A>
User Query: We have a swarm of Honey Bees who just made a comb on a tree branch. Do we need to call someone to come get them to keep them alive through the winter? I know we should try to protect our bees.
Expert Response: Honey bees do not make combs or nests in trees. They need to be inside a structure like a bee box or a hollow tree. It appears to be a swarm of bees. They usually do this in the spring. The old queen from the original nest and half the bees leave. They park themselves somewhere until the bee scouts discover a suitable location. If there is none in the immediate area, the swarm moves on until they locate a new home. The half of the bees remaining in the old nest have a young queen that stays with them. Give this a few days and they will probably be gone. Or if you know a beekeeper, ask if they want them.

Model Output:
{"time_related":False}
<Example 3 End>

<Example 4: Expert mentioned "in early Sept 2022. The lilac began looking bad starting in the summer.", used the time information>
User Query: I have a diseased lilac which may be on its last legs. I have attached some pictures taken in early Sept 2022. The lilac began looking bad starting in the summer. I did not notice any bugs. I am wondering if it is a fungus of some kind. Is there a treatment I can apply this spring to try to save this lilac? I appreciate any information you can provide.
Expert Response: Your Lilac appears to be infected with a fungal pathogen Pseudocercospora leaf spot. Additionally, a disease that causes very similar symptoms is the bacteria Pseudomonas spp. 

Model Output:
{"time_related":True}
<Example 4 End>

<Example 5: Expert mentioned "this year", but not knowing the exact user's time  would not affect the content of the Q&A>
User Query: I have four large deciduous type bushes that I’ve had in my yard for well over 20 years. This year I noticed several dead branches on all four of the plants. As I was trying to do a video to show a friend who majored in horticulture, I shook one of the branches and a billowing cloud of powdery substance came out. It happened throughout all four plants. Can you help me identify what is going on?
Expert Response: You mention deciduous shrubs, but the photos show an evergreen, so we presume you were referring to those, which look like Junipers. There are dozens of juniper varieties, so it’s hard to pinpoint the exact type, though their shape and color resemble 'Gray Owl' or 'Angelica Blue'.

Model Output:
{"time_related":False}
<Example 5 End>

<Example 6: Expert mentioned " starts its spring growth - early March", but not knowing the exact user's time would not affect the content of the Q&A>
ID: #709550
User Query: Hi, I have a 5-8 year old healthy vining hydrangea and the original trellises are falling apart but the tree is pretty wrapped around the trellises. The thicker branches break pretty easily when I try to move them. I need to come up with a plan to redo the trellis but am not sure what to do with the plant and what time of year is best for a project like this. The plant easily extends 15 feet wide and 12 feet high, I don't trim it much. Thank you!
Expert Response: Hello, Many shrubs and vines can take the drastic pruning method called rejuvenation pruning. It is done in the early spring so that the plant has a full season to grow. It involves cutting the shrub or vine back close to the ground. In your case, I would suggest cutting it down to the point where you can work near it to build the new trellis but keeping as much of the plant as you can to allow the rebuild. I would leave it alone for the remainder of this season in order to allow it to store the energy it will need to get started next spring. That stored energy will be key to helping the plant restart after you have pruned it down. Pruning should be done before the plant starts its spring growth - early March. I hope this information helps.
Model Output:
{"time_related":False}
<Example 6 End>

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
            item["time_related"] = response.time_related
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["time_related"] = None
            
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
                        if "time_related" in item and item["time_related"] is not None:
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
                    if "time_related" in item and item["time_related"] is not None:
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

    checker = CheckTimeRelated(args.input_file, args.output_file, args.model_name, args.num_processes)
    checker.extract()
