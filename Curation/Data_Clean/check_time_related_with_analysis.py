import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from chat_models.Gemini import Gemini
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

# Define a new Pydantic model for the direct answer check
class Judge(BaseModel):
    analysis: str
    time_related: bool
    
    def to_json(self):
        return {"analysis": self.analysis, "time_related": self.time_related}

class CheckTimeRelated:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is satisfactory
    def get_prompt(self, item):
        prompt = """I am processing an agricultural Q&A dataset. Our dataset was crawled from a website where experts can see the timestamp of users' questions when answering them, but this timestamp is not included in the users' questions text. I need to determine if the expert used this timestamp information in their answer in a way that would affect the Q&A content.

You should first analyze the Q&A pair and then judge wether the QA is time_related.
Your output should in JSON format, {"analysis":..., "time_related":True/False}

An answer is TIME-RELATED (True) when:
1. The expert refers to the current season, month, or time of year in a way that's not mentioned in the user's question
2. The expert mentions specific time-related conditions (e.g., "this drought," "recent frost," "current weather patterns") not referenced by the user
3. The expert gives timing-specific advice that depends on when the question was asked (e.g., "it's too late this year")
4. The expert refers to recent events or conditions by using phrases like "this year," "recently," "currently," "right now," etc. in a way that's critical to their response
5. The expert contrasts current conditions with past ones (e.g., "this season has been wetter than last year")

An answer is NOT TIME-RELATED (False) when:
1. The expert mentions seasons or timing in general agricultural advice that applies regardless of when the question was asked
2. The expert refers to plant life cycles or stages without specifically tying them to the current time
3. The expert mentions "spring," "summer," etc. as part of general guidance about when certain actions should be taken
4. The expert discusses typical seasonal patterns without indicating they're referring to current conditions
5. The user has already mentioned time-specific information in their question, and the expert simply references this information
6. Experts who do not know the current time can give the same answer

Examples:

<Example 1>
User Query: I found this spider at a property. Can anyone help identify it?
Expert Response: Hello, this appears to be a variation of the orb weaver spider. They are a beneficial spider that clean up the garden of insects. They are usually hesitant to bite - although like all spiders, if instigated, they will bite. They only live a season and usually die around this time of year. Let me know if you have any further questions :)

Model Output:
{
"analysis": "The expert states 'They only live a season and usually die around this time of year' which indicates they're using knowledge of the current time/season to provide this information. Without knowing when the question was asked, this part of the answer wouldn't make sense.",
"time_related: True
}
</Example 1>

<Example 2>
User Query: My tree tops are dying. There are about four dead/dying trees in a bunch of about six. Is this from a drier than normal summer and the trees could not survive at their current density? I do not see any signs of insect infestation but may have missed something. Should I go ahead and cut these trees down and consider them dead?
Expert Response: Yes, I suspect the dry conditions of the past summer - really the past two years have been very dry - have resulted in the dieback and death of these trees. There may also be bugs or disease involved but the underlying reason is likely to be moisture stress.

Model Output:
{
"analysis": "The expert references 'the past summer" and "the past two years have been very dry' which indicates they're using knowledge of recent weather conditions tied to when the question was asked. This temporal context is critical to their diagnosis.",
"time_related: True
}
</Example 2>

<Example 3>
User Query: We have a swarm of Honey Bees who just made a comb on a tree branch. Do we need to call someone to come get them to keep them alive through the winter? I know we should try to protect our bees.
Expert Response: Honey bees do not make combs or nests in trees. They need to be inside a structure like a bee box or a hollow tree. It appears to be a swarm of bees. They usually do this in the spring. The old queen from the original nest and half the bees leave. They park themselves somewhere until the bee scouts discover a suitable location. If there is none in the immediate area, the swarm moves on until they locate a new home. The half of the bees remaining in the old nest have a young queen that stays with them. Give this a few days and they will probably be gone. Or if you know a beekeeper, ask if they want them.

Model Output:
{
"analysis": "Although the expert mentions 'in the spring,' this is describing the typical behavior of bees in general, not specifically relating to the current time. The advice about waiting a few days would apply regardless of when the question was asked.",
"time_related": False
}
</Example 3>

<Example 4>
User Query: I have a diseased lilac which may be on its last legs. I have attached some pictures taken in early Sept 2022. The lilac began looking bad starting in the summer. I did not notice any bugs. I am wondering if it is a fungus of some kind. Is there a treatment I can apply this spring to try to save this lilac? I appreciate any information you can provide.
Expert Response: Your Lilac appears to be infected with a fungal pathogen Pseudocercospora leaf spot. Additionally, a disease that causes very similar symptoms is the bacteria Pseudomonas spp.

Model Output:
{
"analysis": "The user mentions specific times ('early Sept 2022,' 'the summer,' 'this spring') and the expert appears to be responding with this temporal context in mind. The diagnosis is related to the timeline provided.",
"time_related": True
}
</Example 4>

<Example 5>
User Query: I have four large deciduous type bushes that I've had in my yard for well over 20 years. This year I noticed several dead branches on all four of the plants. As I was trying to do a video to show a friend who majored in horticulture, I shook one of the branches and a billowing cloud of powdery substance came out. It happened throughout all four plants. Can you help me identify what is going on?
Expert Response: You mention deciduous shrubs, but the photos show an evergreen, so we presume you were referring to those, which look like Junipers. There are dozens of juniper varieties, so it's hard to pinpoint the exact type, though their shape and color resemble 'Gray Owl' or 'Angelica Blue'.

Model Output:
{
"analysis": "Although the user mentions 'this year,' the expert's response is focused on plant identification and does not reference the timing in a way that affects the content of their answer. The identification would be the same regardless of when the question was asked.",
"time_related": False
}
</Example 5>

<Example 6>
User Query: Hi, I have a 5-8 year old healthy vining hydrangea and the original trellises are falling apart but the tree is pretty wrapped around the trellises. The thicker branches break pretty easily when I try to move them. I need to come up with a plan to redo the trellis but am not sure what to do with the plant and what time of year is best for a project like this. The plant easily extends 15 feet wide and 12 feet high, I don't trim it much. Thank you!
Expert Response: Hello, Many shrubs and vines can take the drastic pruning method called rejuvenation pruning. It is done in the early spring so that the plant has a full season to grow. It involves cutting the shrub or vine back close to the ground. In your case, I would suggest cutting it down to the point where you can work near it to build the new trellis but keeping as much of the plant as you can to allow the rebuild. I would leave it alone for the remainder of this season in order to allow it to store the energy it will need to get started next spring. That stored energy will be key to helping the plant restart after you have pruned it down. Pruning should be done before the plant starts its spring growth - early March. I hope this information helps.

Model Output:
{
"analysis": "Although the expert mentions specific times ('early spring,' 'remainder of this season,' 'early March'), this is general advice about the plant life cycle that would apply regardless of when the question was asked. The guidance is about timing relative to plant growth cycles, not the specific date when the question was submitted.",
"time_related": False
}
</Example 6>

<Example 7>
User Query: What is eating my tomato plants? I just noticed these holes appearing over the last few days.
Expert Response: What you're seeing is likely tomato hornworm damage. This is very common right now as we're in peak hornworm season. Look for large green caterpillars on the undersides of leaves, typically in the evening. Given how late it is in the growing season, if your plants are already producing fruit, you might just want to hand-pick the hornworms rather than spraying.

Model Output:
{
"analysis": "The expert mentions 'right now,' 'peak hornworm season,' and 'how late it is in the growing season' - all indicating they're using knowledge of the current time to provide specific seasonal advice that would change if asked at a different time of year.",
"time_related": True
}
</Example 7>

<Example 8>
User Query: Can I plant asparagus now?
Expert Response: Asparagus should be planted in early spring when the soil can be worked. Plant one-year-old crowns about 12-18 inches apart in trenches that are 6-8 inches deep. Cover with 2 inches of soil initially, then gradually fill in the trench as the plants grow. Asparagus needs full sun and well-draining soil with a pH between 6.5-7.5 for best results. Don't harvest in the first season to allow the plants to establish.

Model Output:
{
"analysis": "Although the user asks about timing ('now'), the expert provides general information about when and how to plant asparagus that would be the same regardless of when the question was asked. The expert describes the proper planting time based on soil conditions, not based on the current date.",
"time_related": False
}
</Example 8>

<Example 9>
User Query: My lilacs have had powered mildew for about 3 summers. They did not bloom at all this year.  My soul tested at ph 7.0. The phosphorus was sufficient but potassium was depleted as was nitrogen. If I treat the plant with fertilizer will it make it stronger for the spring?  Or should I chop them to the base and start again with them.  They are really old I think, I’ve been Sri g for them for 10 years and a decade ago they were already 6 feet tall.
Expert Response: Hello,\nI’m sorry to hear about your lilac not flowering.  Think first about the conditions necessary for a lilac to thrive.  They like alkaline soil, sun and good air movement through the shrub. Your lilac is in alkaline soil. However, has the sun and air movement changed?  Has a large tree nearby shaded it from the sun?  Has a fence, hedge or wall stopped air circulation?  Of course, if the tree is full of old branches, air circulation may have been affected, leading to the development of powdery mildew. It’s often difficult to improve the amount of sunlight the lilac gets if an obstacle has been erected.  The only way to correct that is to move it.  You can prune to correct air circulation.  If you decide to prune, remove the deadwood first, then a few of crossing/rubbing and wrong way branches. You can then remove the oldest and tallest branches first to about 6 inches to one foot which allows suckers to grow up into replacement branches. This will open up the shrub and allow air movement. The ideal time to prune is right after the lilac blooms (or when it should have). If you prune later you may cut off  next year’s flower buds. Resist pruning in the late summer to fall because it may cause a growth spurt that could be damaged by frost.  (You can prune in midwinter but the plant may not bloom the following spring.) For the same reason, you generally don’t fertilize in the late summer/fall.  You don’t want to encourage growth as we move into colder weather.  \nGood luck with your lilac.

{
  "analysis": "The expert’s answer uses common seasonal expressions such as 'right after the lilac blooms,' 'late summer to fall,' and 'midwinter' which are standard horticultural guidelines. These terms refer to general seasonal practices and do not indicate that the expert relied on knowing the user’s current time. The advice applies universally, regardless of when the question was asked.",
  "time_related": False
}
</Example 9>

<Example 10>
User Query: I want to kill “stilt grass” that is growing heavily in my pachysandra garden. What can I do? Please give me some options.  It is too difficult to pull out the grass by hand.
Expert Response: Since pachysandra spreads by runners or \"rhizomes\" and stiltgrass spreads by seed, you could apply a pre-emergent herbicide before the stiltgrass emerges in the spring in order to suppress germination. Apply a preemergent without nitrogen fertilizer. Look for the active ingredient: Prodiamine (Barricade) or other preemergents labeled for crabgrass control. Apply in early spring (March) before it germinates. It germinates earlier than crabgrass so to prevent Japanese stiltgrass the preemergent needs to be applied a couple of weeks earlier than for crabgrass prevention. See these links for more details on stilt grass and its control.

{
  "analysis": "The expert’s advice specifies applying a pre-emergent herbicide in early spring—timing that corresponds with the general growth cycle of stiltgrass. This guidance is standard seasonal advice rather than being tailored to the user’s current time, meaning expert can give the recommendation even though he does not know current time.",
  "time_related": False
}
</Example 10>

<Example 11>
User Query: Hi, the last 2 years I planted pansies in my rock garden together with other annulas and perennials. This year, a little later than usual because of our cold May. Both years, in the days after transplanting, the pansies seemed to dry out very quickly and I often had to water them twice a day. They never started growing, just slowly withered away and died. The other plants grew fine. In the picture, you can see the empty spot between the marigolds and the verbenas. The soil in the garden could be better but each year, I mix in store bought compost and a little sheep/peet where I plant the annuals. What should I do differently? I love pansies!   -Mats
Expert Response: Pansies do not like heat and full sun where as Marigolds and Verbena are quite heat tolerant and thrive in full sun. if you want to plant Pansies, i suggest a slightly shady spot, they will look beautiful as the weather cools and may survive the first frost!

{
  "analysis": "The expert’s answer provides general seasonal guidance by stating that pansies do not tolerate heat and full sun, and that they will thrive as the weather cools and around the first frost. These recommendations are standard horticultural practices that apply regardless of the user's current timestamp, and do not depend on the expert having access to the specific time when the question was asked.",
  "time_related": False
}
</Example 11>

Please judge the following Q&A pair:
"""

        sample_prompt = f"User Query: {item['question']}\nExpert Response: {item['answer']}\n\nModel Output:\n"
        return {"prompt": prompt + sample_prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[])
        elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash" or self.model_name == "gemini-2.0-flash":
            client = Gemini(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
                response = client.chat(prompt=prompt["prompt"], response_format=Judge)
                item["time_related"] = response.time_related
                item["time_related_analysis"] = response.analysis
            elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash" or self.model_name == "gemini-2.0-flash":
                response = client.chat(prompt=prompt["prompt"], response_format=Judge)
                response = json.loads(response)
                item["time_related"] = response["time_related"]
                item["time_related_analysis"] = response["analysis"]
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
