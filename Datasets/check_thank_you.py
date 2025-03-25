import sys
sys.path.append('../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

# Define a new Pydantic model for the direct answer check
class Judge(BaseModel):
    thank_you: bool
    
    def to_json(self):
        return {"thank_you": self.thank_you}

class CheckThankYou:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the user response is just a "Thank you"
    def get_prompt(self, item):
        prompt = """The conversation below contains dialogue between a user and an expert (it may only contain the user). You need to determine whether the conversation is simply the user thanking the expert and the expert responding with "You're welcome." If it is, output "True." If the user asks a new question or provides new information, or if the expert offers new advice, output "False."

Please read the following examples carefully and use them as a basis for your judgment.

<Example 1: User gives additional information, expert responds with new advice>
User: Hi there! Thanks for getting back to me so quickly. The pictures I sent you were watermelons and cucumber. The broccoli is being munched on as well. This is my parents garden and I planted the watermelon on Saturday, and when I returned yesterday the watermelon had been eaten quite significantly. I believe their tiny spinach are being munched on as well. I haven’t seen any pests on the leaves but I will look again this afternoon when I go over there. On the deck she has six big pots with tomatoes in them and I have seen aphids but the leaves of the tomatoes don’t have holes like the plants in the raised beds. I made a solution of diluted neem oil and dish soap and have sprayed everything down several times but the little buggers keep coming back. I feel like it’s two separate pests if that makes sense. The aphids aren’t creating big holes on the tomatoes. I did see some earwigs while I was preparing the soil in the raised beds and I’ve seen a few aphids on the broccoli but not many. We got marigolds and planted them EVERYWHERE a couple weeks ago because my mom read that aphids don’t like them but they don’t appear to be much help so far. The unaffected plants appear to be (for the most part) onions, sugar snap peas, zucchini and blueberries. I could be mistaken but I don’t think those ones have been munched on. I know they have squirrels in the area as well as slugs and snails. There are just a zillion things it could be and I don’t even know where to begin. Thank you for your help! I really appreciate it. TristaOn Tue, May 18, 2021 at 12:31 PM Ask Extension <<personal data hidden>> wrote:
Expert: Thank you.  Slugs typically only feast on the edges of the leaves, and don't leave holes between the veins.  Neem oil works (without dish soap) by cutting off air to the insects, so must be sprayed on them directly, and if they are underneath the leaf, spraying it on top doesn't do anything.  Here is a list of the insects known to be pests of the cucurbit family:  Quick find - Insect crop pests | Pacific Northwest Pest Management Handbooks (pnwhandbooks.org)The articles available through the above pest management overview have photos of the insects, occasionally at different life cycles. "Bugs" suck chlorophyll out of the leaf cells, leaving them yellow.  Most suck from the underside of the leaf.  Beetles chew both on the leaf margins and occasionally within the leaf.  Aphids suck on the leaves (and other tissue) and leave honeydew, which attracts ants.  You can see photos of the eggs, but they aren't doing any harm to the plant until they mature so they can either pierce and suck, or chew on.  Hope this is helpful.

Model Output: 
{"thank_you":False}
<Example 1 End>

<Example 2: User just says "Thanks".>
User: Thanks!!!   From: ask=<personal data hidden> <ask=<personal data hidden>> On Behalf Of Ask Extension Sent: Tuesday, September 19, 2023 3:07 PM To: Trailer, Loris <<personal data hidden>> Subject: Re: Please identify this plant. (#0120440)

Model Output:
{"thank_you":True}
<Example 2 End>

<Example 3: User asks a follow-up question, expert provides new advice>
User: Thank you for your reply! It looks like the fruit will still be safe to eat? On Fri, Apr 26, 2024 at 1:16 PM Ask Extension <<personal data hidden>> wrote:
Expert: Yes. Slime mold is not toxic to humans.  Those with allergies may have a reaction to the mold.  Brush and wash it off thoroughly.

Model Output:
{"thank_you":False}
<Example 3 End>

<Example 4: User says "Thank you" and expert responds with "You're welcome.">
User: Thank you!On Tue, May 21, 2024, 7:05 PM Ask Extension <<personal data hidden>> wrote:
Expert: You’re welcome.

Model Output:
{"thank_you":True}
<Example 4 End>

<Example 5: User just says "Thanks".>
User: Thank you so much for your help. The articles were very helpful too!

Model Output:
{"thank_you":True}
<Example 5 End>

Please judge the following conversation based on the examples above.
"""

        sample_prompt = f"{item['check_thank']}\n\nModel Output:\n"
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
            item["thank_you"] = response.thank_you
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["thank_you"] = None
            
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
                        if "thank_you" in item and item["thank_you"] is not None:
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
                    if "thank_you" in item and item["thank_you"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if user responses just a Thank you.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    checker = CheckThankYou(args.input_file, args.output_file, args.model_name, args.num_processes)
    checker.extract()
