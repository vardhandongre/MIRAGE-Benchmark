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
        prompt = """I am processing an agricultural Q&A dataset. Our dataset was crawled from a website where experts can see the timestamp of users' questions when answering them, but this timestamp is not included in the users' questions text. I need to determine if the expert used this timestamp information in their answer in a way that would affect the Q&A content.

After analyzing each Q&A pair, please respond ONLY with:
{"time_related": True} - if the expert clearly referenced the timestamp in a way that affects the content
{"time_related": False} - if the expert did not use the timestamp or if knowing the exact time would not significantly affect the Q&A content

An answer is TIME-RELATED (True) when:
1. The expert refers to the current season, month, or time of year in a way that's not mentioned in the user's question
2. The expert mentions specific time-related conditions (e.g., "this drought," "recent frost," "current weather patterns") not referenced by the user
3. The expert gives timing-specific advice that depends on when the question was asked (e.g., "it's too late this year, but next spring...")
4. The expert refers to recent events or conditions by using phrases like "this year," "recently," "currently," "right now," etc. in a way that's critical to their response
5. The expert contrasts current conditions with past ones (e.g., "this season has been wetter than last year")

An answer is NOT TIME-RELATED (False) when:
1. The expert mentions seasons or timing in general agricultural advice that applies regardless of when the question was asked
2. The expert refers to plant life cycles or stages without specifically tying them to the current time
3. The expert mentions "spring," "summer," etc. as part of general guidance about when certain actions should be taken
4. The expert discusses typical seasonal patterns without indicating they're referring to current conditions
5. The user has already mentioned time-specific information in their question, and the expert simply references this information
6. The expert mentions time information but knowing the exact time would not change the advice given

Examples:

<Example 1>
User Query: I found this spider at a property. Can anyone help identify it?
Expert Response: Hello, this appears to be a variation of the orb weaver spider. They are a beneficial spider that clean up the garden of insects. They are usually hesitant to bite - although like all spiders, if instigated, they will bite. They only live a season and usually die around this time of year. Let me know if you have any further questions :)

Judgment: {"time_related": True}
Reason: The expert states "They only live a season and usually die around this time of year" which indicates they're using knowledge of the current time/season to provide this information. Without knowing when the question was asked, this part of the answer wouldn't make sense.
</Example 1>

<Example 2>
User Query: My tree tops are dying. There are about four dead/dying trees in a bunch of about six. Is this from a drier than normal summer and the trees could not survive at their current density? I do not see any signs of insect infestation but may have missed something. Should I go ahead and cut these trees down and consider them dead?
Expert Response: Yes, I suspect the dry conditions of the past summer - really the past two years have been very dry - have resulted in the dieback and death of these trees. There may also be bugs or disease involved but the underlying reason is likely to be moisture stress.

Judgment: {"time_related": True}
Reason: The expert references "the past summer" and "the past two years have been very dry" which indicates they're using knowledge of recent weather conditions tied to when the question was asked. This temporal context is critical to their diagnosis.
</Example 2>

<Example 3>
User Query: We have a swarm of Honey Bees who just made a comb on a tree branch. Do we need to call someone to come get them to keep them alive through the winter? I know we should try to protect our bees.
Expert Response: Honey bees do not make combs or nests in trees. They need to be inside a structure like a bee box or a hollow tree. It appears to be a swarm of bees. They usually do this in the spring. The old queen from the original nest and half the bees leave. They park themselves somewhere until the bee scouts discover a suitable location. If there is none in the immediate area, the swarm moves on until they locate a new home. The half of the bees remaining in the old nest have a young queen that stays with them. Give this a few days and they will probably be gone. Or if you know a beekeeper, ask if they want them.

Judgment: {"time_related": False}
Reason: Although the expert mentions "in the spring," this is describing the typical behavior of bees in general, not specifically relating to the current time. The advice about waiting a few days would apply regardless of when the question was asked.
</Example 3>

<Example 4>
User Query: I have a diseased lilac which may be on its last legs. I have attached some pictures taken in early Sept 2022. The lilac began looking bad starting in the summer. I did not notice any bugs. I am wondering if it is a fungus of some kind. Is there a treatment I can apply this spring to try to save this lilac? I appreciate any information you can provide.
Expert Response: Your Lilac appears to be infected with a fungal pathogen Pseudocercospora leaf spot. Additionally, a disease that causes very similar symptoms is the bacteria Pseudomonas spp.

Judgment: {"time_related": True}
Reason: The user mentions specific times ("early Sept 2022," "the summer," "this spring") and the expert appears to be responding with this temporal context in mind. The diagnosis is related to the timeline provided.
</Example 4>

<Example 5>
User Query: I have four large deciduous type bushes that I've had in my yard for well over 20 years. This year I noticed several dead branches on all four of the plants. As I was trying to do a video to show a friend who majored in horticulture, I shook one of the branches and a billowing cloud of powdery substance came out. It happened throughout all four plants. Can you help me identify what is going on?
Expert Response: You mention deciduous shrubs, but the photos show an evergreen, so we presume you were referring to those, which look like Junipers. There are dozens of juniper varieties, so it's hard to pinpoint the exact type, though their shape and color resemble 'Gray Owl' or 'Angelica Blue'.

Judgment: {"time_related": False}
Reason: Although the user mentions "this year," the expert's response is focused on plant identification and does not reference the timing in a way that affects the content of their answer. The identification would be the same regardless of when the question was asked.
</Example 5>

<Example 6>
User Query: Hi, I have a 5-8 year old healthy vining hydrangea and the original trellises are falling apart but the tree is pretty wrapped around the trellises. The thicker branches break pretty easily when I try to move them. I need to come up with a plan to redo the trellis but am not sure what to do with the plant and what time of year is best for a project like this. The plant easily extends 15 feet wide and 12 feet high, I don't trim it much. Thank you!
Expert Response: Hello, Many shrubs and vines can take the drastic pruning method called rejuvenation pruning. It is done in the early spring so that the plant has a full season to grow. It involves cutting the shrub or vine back close to the ground. In your case, I would suggest cutting it down to the point where you can work near it to build the new trellis but keeping as much of the plant as you can to allow the rebuild. I would leave it alone for the remainder of this season in order to allow it to store the energy it will need to get started next spring. That stored energy will be key to helping the plant restart after you have pruned it down. Pruning should be done before the plant starts its spring growth - early March. I hope this information helps.

Judgment: {"time_related": False}
Reason: Although the expert mentions specific times ("early spring," "remainder of this season," "early March"), this is general advice about the plant life cycle that would apply regardless of when the question was asked. The guidance is about timing relative to plant growth cycles, not the specific date when the question was submitted.
</Example 6>

<Example 7>
User Query: What is eating my tomato plants? I just noticed these holes appearing over the last few days.
Expert Response: What you're seeing is likely tomato hornworm damage. This is very common right now as we're in peak hornworm season. Look for large green caterpillars on the undersides of leaves, typically in the evening. Given how late it is in the growing season, if your plants are already producing fruit, you might just want to hand-pick the hornworms rather than spraying.

Judgment: {"time_related": True}
Reason: The expert mentions "right now," "peak hornworm season," and "how late it is in the growing season" - all indicating they're using knowledge of the current time to provide specific seasonal advice that would change if asked at a different time of year.
</Example 7>

<Example 8>
User Query: Can I plant asparagus now?
Expert Response: Asparagus should be planted in early spring when the soil can be worked. Plant one-year-old crowns about 12-18 inches apart in trenches that are 6-8 inches deep. Cover with 2 inches of soil initially, then gradually fill in the trench as the plants grow. Asparagus needs full sun and well-draining soil with a pH between 6.5-7.5 for best results. Don't harvest in the first season to allow the plants to establish.

Judgment: {"time_related": False}
Reason: Although the user asks about timing ("now"), the expert provides general information about when and how to plant asparagus that would be the same regardless of when the question was asked. The expert describes the proper planting time based on soil conditions, not based on the current date.
</Example 8>

Please judge the following Q&A pair:
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
