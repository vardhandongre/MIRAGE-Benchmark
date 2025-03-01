import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

class QA(BaseModel):
    question: str
    answer: str    
    def to_json(self):
        return {"question": self.question, "answer": self.answer}

class Reformat:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):        
        prompt = """\
I am processing an agricultural Q&A dataset that was scraped from a website where users post questions and experts provide answers. Some parts of these conversations contain expressions that should be removed or modified to make the Q&A pairs suitable for training a large language model. For instance, phrases like "Thank you for contacting Ask Extension." (i.e., inappropriate greetings) need to be removed. !!Do not change other content within the Q&A!!
Your task is to review each Q&A pair and determine if any modifications are necessary. If modifications are needed, output a JSON object in the following format:
{"question": "<modified question>", "answer": "<modified answer>"}
If no modifications are required, output:
{"question": null, "answer": null}

Please read the following examples carefully and use them as a basis for your output:

<Example 1>
User Query: Recently noticed many leaves are developing these spots. How can I save this plant?
Expert Response: Thanks for using ask an Expert, but as we talked the other day, too much sunlight is likely the culprit of the leaf issue.

Model Output:
{"question": "Recently noticed many leaves are developing these spots. How can I save this plant?", "answer": "Too much sunlight is likely the culprit of the leaf issue."}
<Example 1 End>

<Example 2>
User Query: I know that some loss of needles is normal, but I’m concerned at the amount here.
Expert Response: Hi, Thanks for reaching out to Ask Extension. I am happy to help! Hmm. From your pictures, I do not see the needle loss you are seeing. I do see some browning on your beautiful giant sequoia.

Model Output:
{"question": "I know that some loss of needles is normal, but I’m concerned at the amount here.", "answer": "From your pictures, I do not see the needle loss you are seeing. I do see some browning on your beautiful giant sequoia. As you mentioned, this sort of browning and needle loss, or needle drop, is typical this time of year."}
<Example 2 End>

<Example 3>
User Query: Hi, does anyone know what's the name of this plant? It has purplish flower and the leaves have a sort of texture.
Expert Response: Hi, that looks like Plectranthus 'Mona Lavender.' It is perennial in Zone 10, and grown as an annual in colder climates. Most Plectranthus are grown for their attractive foliage, rather than flowers, but 'Mona Lavender' has showy flowers that bloom for much of the summer.

Model Output:
{"question": "What's the name of this plant? It has purplish flower and the leaves have a sort of texture.", "answer": "That looks like Plectranthus 'Mona Lavender.' It is perennial in Zone 10, and grown as an annual in colder climates. Most Plectranthus are grown for their attractive foliage, rather than flowers, but 'Mona Lavender' has showy flowers that bloom for much of the summer."}
<Example 3 End>

<Example 4>
User Query: My uncle has this vine climbing his trees and also populating the ground. I have never seen it before and am curious about the details of it and control measures. Thank you!
Expert Response: Thanks for your message. Here’s a response from a renown horticulturist regarding your question: This is snailseed vine, a common invader in the eastern half of the state. It’s actually attractive when observed growing on an old barbed wire fence in nature, but when it starts to take over a planting of landscape shrubs it’s a different matter. 

Model Output:
{"question": "There is a vine climbing trees and spreading on the ground that I have never seen before. I am curious about its characteristics and the recommended control measures.", "answer": "This is snailseed vine, a common invader in the eastern half of the state. It’s actually attractive when observed growing on an old barbed wire fence in nature, but when it starts to take over a planting of landscape shrubs it’s a different matter. It’s a perennial woody vine that forms a thick root system. If you can get to the crown of its roots you can dig it out fairly quickly when the soil is moist, but otherwise, the best luck is with a roll-type mulch put beneath shrubs and overlapped by 4 or 5 inches so the vine can’t grow up through the seams. The roll mulch can be concealed with finely ground pine bark mulch for aesthetics, but shredded tree leaves or compost could work just as well. As pretty as the fruit is, it’s advisable not to have the plant in your landscape. The fruit only lasts for a few days before birds consume it, leading to the plant spreading rapidly."}
<Example 4 End>

<Example 5>
User Query: I planted Creeping Jenny in my garden with the hope of not having to use so much mulch in the future. It has gotten quite thick and is crowding my plants. I have a couple that in that spot that haven't come back this spring - not sure if it is from the winter or the groundcover. Is this okay, or do I need to pull it out before it chokes out my perennials?
Expert Response: Hello. I am a Master Gardener. I have researched your question at various Extension services. Creeping Jenny can be very invasive in a garden and will crowd out other plants you may want to keep. Yes, it is a good idea to pull it out before it chokes out your other perennials. Thank you for your inquiry.

Model Output:
{"question": "I planted Creeping Jenny in my garden with the hope of not having to use so much mulch in the future. It has gotten quite thick and is crowding my plants. I have a couple that in that spot that haven't come back this spring - not sure if it is from the winter or the groundcover. Is this okay, or do I need to pull it out before it chokes out my perennials?", "answer": "Creeping Jenny can be very invasive in a garden and will crowd out other plants. It is advisable to pull it out before it chokes out your perennials."}
<Example 5 End>

<Example 6>
User Query: I have no idea why my Rudbeckia fulgida seems to be having a problem. I had a previous variety several years ago and it was pretty bad off with multiple issues. The leaves would get purple spots and die back to crunchy brown. I pulled them all up and planted other plants mostly natives although there’s a purple coneflower that seems ok. About 8 ft away on the other side of the walkway I planted Rudbeckia fulgida and other natives. This plant is about 2 years old. Several stems are dying back and there are brown spots. I have pets but haven’t seen them bothering this plant. Would you have any ideas on the problem and if it’s fixable?
Expert Response: Dying individual or scattered stems in a mass planting or perennial clump can wilt or be killed for a variety of reasons that are sometimes hard to diagnose. Stem-boring insect larvae, breakage due to wildlife, and certain infections (Southern Blight being one of several) can all result in stem collapse and leaf browning.

Model Output:
{"question": null, "answer": null}
<Example 6 End>

<Example 7>
User Query: Moved into a new house and the dogwood tree is missing bark on the trunk where it meets the ground. It initially had grass growing up around the trunk and after removing that and mulching, I found some fungus that I removed. They looked like some kind of mushroom, but I can't find a picture now. It was also looking thirsty from the hot dry summer so I've been watering with a dripping 5 gallon bucket a few times a week, but it still has many wilted leaves. Only a few berries ripened to red but there are a good number of green berries still on. What can I do to protect the exposed trunk and should I be watering significantly more?
Expert Response: Thanks for the question. You are doing the right thing in watering during hot dry weather. The damaged bark is a bad sight! As long as there is some continuous bark from the ground on up the trunk to the upper branches there is hope that the tree will survive. Do be very careful not to do any more damage. 

Model Output:
{"question": "Moved into a new house and the dogwood tree is missing bark on the trunk where it meets the ground. It initially had grass growing up around the trunk and after removing that and mulching, I found some fungus that I removed. They looked like some kind of mushroom, but I can't find a picture now. It was also looking thirsty from the hot dry summer so I've been watering with a dripping 5 gallon bucket a few times a week, but it still has many wilted leaves. Only a few berries ripened to red but there are a good number of green berries still on. What can I do to protect the exposed trunk and should I be watering significantly more?", "answer": "You are doing the right thing in watering during hot dry weather. The damaged bark is a bad sight! As long as there is some continuous bark from the ground on up the trunk to the upper branches there is hope that the tree will survive. Do be very careful not to do any more damage."}
<Example 7 End>

Please revise the following Q&A pairs.
"""

        prompt += f"User Query: {item['question']}\nExpert Response: {item['answer']}\n\nModel Output:\n"

        return {"prompt": prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            response = client.chat(prompt=prompt["prompt"], response_format=QA)
            item["new_question"] = response.question
            item["new_answer"] = response.answer
            item["info"] = client.info()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["new_question"] = None
            item["new_answer"] = None
            
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def reformatting(self):
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "new_question" in item and "new_answer" in item and item["new_question"] is not None and item["new_answer"] is not None:
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
                    if "new_question" in item and "new_answer" in item and item["new_question"] is not None and item["new_answer"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if expert answers are direct using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    reformat = Reformat(args.input_file, args.output_file, args.model_name, args.num_processes)
    reformat.reformatting()
