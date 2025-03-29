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
        prompt = """I am processing an agricultural Q&A dataset and need assistance in identifying whether each Q&A pair is location contextual. Please classify each pair as {"location_related": True} or {"location_related": False} based on the following criteria:

CLASSIFY AS LOCATION-RELATED (True) WHEN:
1. The expert mentions a specific location (city, county, state, region, etc.) that wasn't mentioned in the user's question
2. The expert refers to vague locations like "our area," "this region," "your locality," etc. when the user didn't mention location
3. The expert cites location-specific regulations, laws, or practices not referenced by the user
4. The expert gives location-dependent advice without the user providing location context

DO NOT CLASSIFY AS LOCATION-RELATED (False) WHEN:
1. The user has already mentioned the specific location in their question
2. The expert mentions general locations as part of factual knowledge (e.g., "this species is native to the Northeast")
3. References to personal spaces like "my yard," "my garden," "my house," etc. appear in either question or answer
4. The expert discusses habitat or growing conditions without specifying geographic locations
5. The expert is providing general information about the geographic distribution of a species as a matter of expertise

Below are examples demonstrating these distinctions:

<Example 1: Expert mentioned location-specific regulation>
User Question: We recently got some landscaping done at our house and now notice we have North American Oil Beetles in our yard. We have noticed three clumps of them and several scattered in the grass. Our internet research says they can cause painful blisters and I fear we have too many to remove by hand. What should and can be done to remove them? Thanks so much.
Expert Answer: These are not ordinarily a problem in yards. Since these are damaging to bee populations, it would be good to eradicate them or at least bring down the population to the usual insignificant levels. You cannot spray pesticides on your lawn in Montgomery County. We'd recommend that you not try to hand-pick them from the lawn, but simply walk on them and crush them as much as possible. Leave the bodies to decompose. If you have a significant population of them next year, please contact us again. These insects eat vegetative matter, particularly buttercups. Buttercups are fairly easy to pull when the soil is moist. You might want to remove buttercups if you have them around.

Judgment: {"location_related": True}
Reason: The expert mentioned Montgomery County regulations without the user specifying their location.
</Example 1>

<Example 2: User already provided location>
User Question: Hello, I found this plant has aggressively grown on disturbed soil in the area around Larkspur in Douglas County. I have not found it on the noxious weeds list by the County or State and was wondering what the identity is and if I should be getting rid of it.
Expert Answer: It may be a white heath aster; it is hard to tell without the flowers being opened. It is not a noxious weed. One person's flower is another person's weed, so it is up to your needs as to whether you should get rid of it. I believe it will continue to spread as it produces many seeds and will not go away on its own.

Judgment: {"location_related": False}
Reason: The user already specified their location (Larkspur in Douglas County), so the expert didn't introduce new location context.
</Example 2>

<Example 3: Expert mentioned vague location>
User Question: Trying to figure out if this will damage my lilacs. It started on this one and has moved to others nearby. What can I do? My lilacs are 15-19 years old.
Expert Answer: The lichen appearing on your lilacs suggests humidity and slow growth. An older lilac shrub growing slowly is natural and not necessarily a reason for alarm. The lichen is not corrupting the plant on its own. But while the lichen is not causing a problem, it may be a signal that your lilacs are in distress. We have seen a lot of lichen appearing on trees in our area this summer as it was unusually moist in the early months. It also is possible that your tree is in too protected a position where moisture is accumulating and that any thinning of nearby plants that you can do to increase air flow will have a benefit.

Judgment: {"location_related": True}
Reason: The expert mentioned "our area" which introduces location context not provided by the user.
</Example 3>

<Example 4: Personal space reference, not location-specific>
User Question: I get these small mounds of dirt in my yard. I am new to the area and haven't come across this before. I need to identify them so I know how to treat them. It wouldn't be so bad if there weren't so many.
Expert Answer: Sorry for the slow reply. This is likely from earthworms. This event often happens after a good soaking rain. It is nothing to worry about, they are aerating the soil for you. The mounds can be raked down if you don't like the way they look. Hope this is helpful.

Judgment: {"location_related": False}
Reason: The mention of "yard" is a personal space reference, not a geographic location. The expert's answer doesn't introduce specific location information.
</Example 4>

<Example 5: Expert providing factual knowledge about species distribution>
User Question: I spotted this yesterday in my backyard, visiting the boneset flowers. Is this a grizzled skipper? From what I can find, that species is not typically found in my county. It had about a 4cm wingspan. Thank you!
Expert Answer: That is a checkered skipper, Pyrgus communis. It is a common skipper in the southern part of the lower peninsula.

Judgment: {"location_related": False}
Reason: The expert's mention of "southern part of the lower peninsula" is providing factual knowledge about the species' distribution as part of their expertise, not introducing location-specific advice.
</Example 5>

<Example 6: Expert mentioning regional timing information>
User Question: When should I plant my tomato seedlings outdoors?
Expert Answer: In our growing zone, it's best to wait until after the last frost date, typically mid-May. Make sure soil temperatures are consistently above 60°F for best results.

Judgment: {"location_related": True}
Reason: The expert references "our growing zone" and specific timing that is location-dependent without the user specifying their location.
</Example 6>

<Example 7: Expert mentioning regional pest information>
User Question: What are these small holes in my apple tree leaves?
Expert Answer: The damage pattern suggests apple flea weevil, which has become increasingly common in our region over the past three years. Unlike other leaf miners, these beetles create distinctive shot-hole patterns as the damaged tissue falls out.

Judgment: {"location_related": True}
Reason: The expert mentions "our region" and provides region-specific pest information without the user specifying location.
</Example 7>

Please judge the following Q&A pairs:
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
