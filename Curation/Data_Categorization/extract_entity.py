import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
from typing import List


class Entity(BaseModel):
    name: str
    aliases: List[str]

    def to_json(self):
        return {
            "name": self.name,
            "aliases": self.aliases
        }

class EntityOther(BaseModel):
    name: str
    aliases: List[str]
    type: str

    def to_json(self):
        return {
            "name": self.name,
            "aliases": self.aliases,
            "type": self.type
        }


class ExtractEntity:
    def __init__(self, input_file, output_file, model_name="gpt-4o-mini", num_processes=None, plant_only=False):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        # Extract Plant Only
        self.plant_only = plant_only
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):
        if self.plant_only:
            prompt = """\
Extract the correct main plants mentioned in the Q&A pairs. Please try to extract specific plant name, and aliases (all in lowercase and in the singular form). \
Do not invent names!! If no plant name is mentioned, output "".

Please read the following examples carefully and use them as a basis for your output.
<Example 1>
User Question:  What is this tree? This tree grows in a town in Texas behind a local courthouse. Could you please let me know the scientific and common name?
Expert Answer:  That is indeed a Jujube tree (Ziziphus jujuba). It can grow to be a beautiful medium-sized tree in Texas and can be very prolific in producing fruit which are quite tasty. The fruit is commonly referred to as "red dates" when dried. Look around the base of the tree; you should be able to see some suckers which could be dug up and replanted.

Model Output:
{"name": "jujube tree", "aliases": ["ziziphus jujuba"]}
<Example 1 End>

<Example 2>
User Question: Not sure what this flower is
Expert Answer: You are fortunate to have that growing in your yard. That is a Hurricane Lily, also known as a red spider lily. Latin name is probably Lycoris radiata. They are an heirloom bulb, and are uncommon and expensive.

Model Output:
{"name": "hurricane lily", "aliases": ["red spider lily", "lycoris radiata"]}
<Example 2 End>

<Example 3>
User Question: I would like to know the name of my plant
Expert Answer: The plant looks like an Angel Wing Begonia.

Model Output:
{"name": "angel wing begonia", "aliases": []}
<Example 3 End>

<Example 4>
User Query: I never seen a spider like this before and I seen these in 2 different locations in my yard. Currently on a bush near my deck. I looked online and the only comparison shows it being native to Europe and not in my area. Thank you.
Expert Response: Hello! Happy to help. Beautiful photograph! It's called a black and yellow garden spider and it can take care of unwanted pests in our gardens. 

Model Output:
{"name": "", "aliases": []}
<Example 4 End>

Please extract the plant entities from the following Q&A pairs:\n"""                

            prompt += f"User Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Output:\n"
        if not self.plant_only:
            if item.get("category") in [
                "Plant Identification", 
                "Plant Care and Gardening Guidance"
            ]:
                prompt = """\
Extract the correct main plants mentioned in the Q&A pairs. Please try to extract specific plant name, and aliases (all in lowercase and in the singular form). \
Do not invent names!! If no plant name is mentioned, output "".

Please read the following examples carefully and use them as a basis for your output.
<Example 1>
User Question:  What is this tree? This tree grows in a town in Texas behind a local courthouse. Could you please let me know the scientific and common name?
Expert Answer:  That is indeed a Jujube tree (Ziziphus jujuba). It can grow to be a beautiful medium-sized tree in Texas and can be very prolific in producing fruit which are quite tasty. The fruit is commonly referred to as "red dates" when dried. Look around the base of the tree; you should be able to see some suckers which could be dug up and replanted.

Model Output:
{"name": "jujube tree", "aliases": ["ziziphus jujuba"]}
<Example 1 End>

<Example 2>
User Question: Not sure what this flower is
Expert Answer: You are fortunate to have that growing in your yard. That is a Hurricane Lily, also known as a red spider lily. Latin name is probably Lycoris radiata. They are an heirloom bulb, and are uncommon and expensive.

Model Output:
{"name": "hurricane lily", "aliases": ["red spider lily", "lycoris radiata"]}
<Example 2 End>

<Example 3>
User Question: I would like to know the name of my plant
Expert Answer: The plant looks like an Angel Wing Begonia.

Model Output:
{"name": "angel wing begonia", "aliases": []}
<Example 3 End>

<Example 4>
User Query: I never seen a spider like this before and I seen these in 2 different locations in my yard. Currently on a bush near my deck. I looked online and the only comparison shows it being native to Europe and not in my area. Thank you.
Expert Response: Hello! Happy to help. Beautiful photograph! It's called a black and yellow garden spider and it can take care of unwanted pests in our gardens. 

Model Output:
{"name": "", "aliases": []}
<Example 4 End>

Please extract the plant entities from the following Q&A pairs:\n"""                

                prompt += f"User Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Output:\n"
            elif item.get("category") in [
                "Weeds/Invasive Plants Management"
            ]:
                prompt = """\
Extract the correct main weed/invasive plant mentioned in the Q&A pairs. Please try to extract specific plant name, and aliases (all in lowercase and in the singular form). \
Do not invent names!! If no weed/invasive plant name is mentioned, output "".

Please read the following examples carefully and use them as a basis for your output.
<Example 1>
User Question: What is the name of this weed?
Expert Answer: Chamber bitter, Phyllanthus urinaria. Appearance: Phyllanthus urinaria is an erect to prostrate, slender, glabrous herb, 4-14 in. (10-35 cm) high. 

Model Output:
{"name": "chamber bitter", "aliases": ["phyllanthus urinaria"]}
<Example 1 End>

<Example 2>
User Question: I would like to know how to remove a particular weed out of my ground cover.
Expert Answer: I'm afraid the only way to remove invasives like weeds from mature ground cover is by hand. Chemicals will kill the wanted plant along with the weeds. Cultivation by hoe is out because the structure of the ground cover is long soil-level branches, some of which will root along the branch. Get the weeds early so they don't seed and you will have less of a problem in the future.

Model Output:
{"name": "", "aliases": []}
<Example 2 End>

<Example 3>
User Question: I have a plant that is overcoming my yard. This was here when we purchased the home 13 years ago. We have tried to tame it, and it comes back more, and more powerful. It comes up through the driveway. I made a flower bed, and it over ran that. Any idea what it is? And how to get rid of it? Thanks You.
Expert Answer: The plant in your photos is Japanese Knotweed. This is, as you know, a serious invasive plant that is widely spread across the northeast. 

Model Output:
{"name": "japanese knotweed", "aliases": []}
<Example 3 End>

<Example 4>
User Question: I found this plant among my bushes and perennial flowers. It has already expanded to 8-10 plants over a period of a month. Its tiny white flowers are like the catmints I had planted to fill the background gap between my showy perennial flowers. They close and die to tiny seed size with spores that stick to one's cloth. I had not planted this, nor have I ever seen it before. I seriously doubt that the catmints have changed into this on their own! I have dug them out, cut the flower heads, and hidden them in a trash bag, careful not to help them spread.
Expert Answer: This looks like Beggarslice (Hackelia virginiana), also commonly called stickseed or sticktight. It is not an invasive plant but actually is native to Maryland. The nectar of the flowers provides food for some native bees and other insects. However, since you don't want to have it among your perennial flowers, you did the right thing by cutting down the flower stalks and digging it out to reduce its spread.

Model Output:
{"name": "beggarslice", "aliases": ["hackelia virginiana", "stickseed", "sticktight"]}
<Example 4 End>

Please extract the weed/invasive plant entities from the following Q&A pairs:\n"""

                prompt += f"User Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Output:\n"                
            elif item.get("category") in [
                "Insect and Pest Identification", 
                "Insect and Pest Management"
            ]:
                prompt = """Extract the correct main insect and pest mentioned in the question-answer pair. Please try to extract specific insect and pest name, and aliases (all in lowercase and in the singular form). Do not invent names!! If no insect and pest name is mentioned, output "".

Please read the following examples carefully and use them as a basis for your output.

<Example 1>
User Question: These were unmoving in a filbert leaf. Periodically one and then another flew off and then back to the leaf. We have never seen these before.
Expert Answer: You found a group of male longhorn bees hanging out together on a leaf. Males have very long antennae whereas the females have shorter ones. 

Model Output:
{"name": "longhorn bee", "aliases": []}
<Example 1 End>

<Example 2>
User Question: Can you identify what is pushing back all the top layer of grass? We have large sections/ patches all over our grass.
Expert Answer: Good Afternoon, It looks like you may have some skunks and/or raccoons. The usual suspects for yard and garden damage are skunks, raccoons, or moles. I hope this helps. Thanks for using our service.

Model Output:
{"name": "", "aliases": []}
<Example 2 End>

<Example 3>
User Question: What insect is this on my redtwig dogwood leaf?
Expert Answer: This is probably an immature tree cricket.

Model Output:
{"name": "tree cricket", "aliases": []}
<Example 3 End>

<Example 4>
User Question: Caught this bug in my house and wondered if it is a good bug to have around or should I destroy it?
Expert Answer: Hello,  That looks like a Hermit Flower Beetle Grub (Osmoderma eremicola). This is a harmless beetle. 

Model Output:
{"name": "hermit flower beetle grub", "aliases": ["osmoderma eremicola"]}
<Example 4 End>

Please extract the insect and pest entities from the following Q&A pairs:\n"""       
                prompt += f"User Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Output:\n"
            elif item.get("category") in [
                "Plant Disease Identification",
                "Plant Disease Management"
            ]:
                prompt = """\
Extract the correct main plant diseases mentioned in the Q&A pairs. Please try to extract specific plant disease name, and aliases (all in lowercase and in the singular form). \
Do not invent names!! If no plant disease name is mentioned, output "".

Please read the following examples carefully and use them as a basis for your output.

<Example 1>
User Question: Two peach trees have similar disease symptoms on their leaves. Possibly peach leaf curl.
Expert Answer: Yes, definitely peach leaf curl.

Model Output:
{"name": "peach leaf curl", "aliases": []}
<Example 1 End>

<Example 2>
User Question: I have numerous laurels in my yard. Several of the shrubs have developed brown leaves, which then start spreading throughout the shrub, then completely killing the shrub back. Is there a disease attacking laurels or could this be some type of insect? I don't see anything on the leaves.
Expert Answer: That looks like cold/winter damage from this past December. We went into winter dry and warm. Then the temperatures dropped suddenly and low. We are seeing damage on many shrubs, including laurels. The damage is typically limited to the foliage. Once spring comes the plants will put out new foliage.

Model Output:
{"name": "", "aliases": []}
<Example 2 End>

<Example 3>
User Question: Bought from a reputable greenhouse and growing on my deck. I use a moisture meter and water when not too wet and not too dry. Growing in a mountainous area. These fruits were on the plant when purchased. No new flowers or fruits.
Expert Answer: Hello, this looks like catfacing (Cat-facing), which is a very common problem, especially on larger tomatoes and certain varieties. The fruit is perfectly safe to eat, and the fruit just looks a bit misshapen. They should ripen normally.

Model Output:
{"name": "catfacing", "aliases": ["cat-facing"]}
<Example 3 End>

Please extract the plant disease entities from the following Q&A pairs:
""" 
                prompt += f"User Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Output:\n"               
            elif item.get("category") in ["Other", "Others"]:
                prompt = (
                    'Extract the main entity (plant, insect, or disease) mentioned in the question-answer pair. '
                    'If none is mentioned, output "None". '
                    'Present the result in JSON format as follows: {"name": "...", "aliases": [...], "type": "plant" or "insect" or "disease"}\n\n'
                    """
Please read the following examples carefully and use them as a basis for your output. If no entity is mentioned, output {"name": "", "aliases": [], "type": ""}.

<Example 1>
User Question: Two peach trees have similar disease symptoms on their leaves. Possibly peach leaf curl.
Expert Answer: Yes, definitely peach leaf curl.

Model Output:
{"name": "peach leaf curl", "aliases": [], "type": "disease"}
<Example 1 End>

<Example 2>
User Question: What insect is this on my redtwig dogwood leaf?
Expert Answer: This is probably an immature tree cricket.

Model Output:
{"name": "tree cricket", "aliases": [], "type": "insect"}
<Example 2 End>

<Example 3>
User Question: What is the name of this weed?
Expert Answer: Chamber bitter, Phyllanthus urinaria. Appearance: Phyllanthus urinaria is an erect to prostrate, slender, glabrous herb, 4-14 in. (10-35 cm) high. 

Model Output:
{"name": "chamber bitter", "aliases": ["phyllanthus urinaria"]}
<Example 3 End>"""
                    
                    f'Q: {item["question"]}\nA: {item["answer"]}'
                )
            else:
                print(f"Unknown category: {item.get('category')}")
        return {"prompt": prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt_data = self.get_prompt(item)
        
        if model_name.startswith("gpt"):
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{model_name}' not supported.")
    
        try:
            if self.plant_only:
                if item.get("category") in [
                                "Plant Identification", 
                                "Plant Care and Gardening Guidance",
                                "Weeds/Invasive Plants Management"
                            ]:
                    item["plant"] = item["entity"]
                else:
                    response = client.chat(prompt=prompt_data["prompt"], response_format=Entity)
                    item["plant"] = {"name": response.name, "aliases": response.aliases}
            else:
                if item.get("category") in ["Other", "Others"]:
                    response = client.chat(prompt=prompt_data["prompt"], response_format=EntityOther)
                    item["entity"] = {"name": response.name, "aliases": response.aliases, "type": response.type}
                else:
                    response = client.chat(prompt=prompt_data["prompt"], response_format=Entity)
                    item["entity"] = {"name": response.name, "aliases": response.aliases}

            item["info"] = client.info()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            if self.plant_only:
                item["plant"] = None
            else:
                item["entity"] = None
        
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
        return item.get('id')


    def extract(self):
        with open(self.input_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if not self.plant_only and "entity" in item and item["entity"] is not None:
                            processed_ids.add(item['id'])
                        elif self.plant_only and "plant" in item and item["plant"] is not None:
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
            processed_ids = set()
            for line in f:
                try:
                    item = json.loads(line)
                    if not self.plant_only and "entity" in item and item["entity"] is not None:
                        processed_ids.add(item['id'])
                        valid_items.append(item)

                    elif self.plant_only and "plant" in item and item["plant"] is not None:
                        processed_ids.add(item['id'])
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}.")
        print(f"Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entity (plant, insect/pest, or disease) based on category using LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--plant_only", action="store_true", help="Extract plant entity only.")
    args = parser.parse_args()

    extractor = ExtractEntity(input_file=args.input_file, output_file=args.output_file, model_name=args.model_name, num_processes=args.num_processes, plant_only=args.plant_only)
    extractor.extract()