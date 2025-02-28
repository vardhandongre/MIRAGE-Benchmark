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
class Satisfactory(BaseModel):
    satisfactory: bool
    
    def to_json(self):
        return {"satisfactory": self.satisfactory}

class CheckSatisfactory:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is satisfactory
    def get_prompt(self, item):
        # prompt for Real VQA dataset: for links check, and for ask for more question and pictures
        prefix = """\
I am cleaning an agricultural Q&A dataset and need your help to determine if the expert's answer is satisfactory. If the expert's answer is unsatisfactory, such as suggesting to contact someone else, asking for more information, indicating uncertainty, admitting they cannot help, or if the answer contains links or mentions that there is a link here, output {"satisfactory": False}. Otherwise, output {"satisfactory": True}. 
Here are some examples to help you understand the task better:

<Example1: suggesting to contact someone else>
Expert's Answer:
Good morning Dave,  Thank you for contacting AnswerLine. I would recommend contacting the Linn County Master Gardeners for assistance with your question. They can be reached at <personal data hidden> or their Hortline at<personal data hidden> (Monday - Thursday from 10AM-12PM).'
Judgement: 
{"satisfactory": False}
<Example1>

<Example2: links in the answer "txsmartscape.com">
Expert's Answer:
What you are seeing is called Horseherb, Calyptocarpus vialis. This plant is a groundcover that is getting attention as a sustainable lawn. I am attaching a few links that will give you information concerning this plant and how to kill it if you want.  Horseherb Details - Texas SmartScape Plant Database (txsmartscape.com)  Meet Horseherb, the Misunderstood Weed - Garden Style San Antonio  HerbPublicationSCS200713.indd (tamu.edu)
Judgement:
{"satisfactory": False}
<Example2>

<Example3: indicating uncertainty>
Expert's Answer:
We can not tell for sure from your photos what your issue is. Possibilities include wildlife burrowing and stormwater issues.
Judgement: 
{"satisfactory": False}
<Example3>

<Example4: No links but mention a link exists "Here is a link">
Expert's Answer:
What is your plant? It looks like it could be some kind of Aloe. It's fairly close to a plant called Partridge Breast Aloe. It flowers with tall red/orange flowers from the top. Here is a link to some more info about the Partridge Breast Aloe.
Judgement: 
{"satisfactory": False}
<Example4>

<Example5: No links but mention a link exists "Please open the drop down boxes">
Expert's Answer:
I believe that is early Black Knot. Please open the drop down boxes in this link to deal with it ASAP. Black knot | UMN Extension
Judgement: 
{"satisfactory": False}
<Example5>

<Example6: No links but mention a link exists "Here is more information about apple scab">
Expert's Answer:
If you compost be sure your compost reaches temperatures that kill fungi and weed seeds. If not, sending your yard waste to a city processor is the better option. Here is more information about apple scab. Good luck!
Judgement: 
{"satisfactory": False}
<Example5>

<Example7: Ask for more information "take another picture and send it in">
Expert's Answer:
It may not be anything to worry about. Just keep an eye on them. If you notice adult insects and they feed on the leaves of the wild rose, take another picture and send it in. Then we'll have a better idea if you need to use an insecticide or not.
Judgement:
{"satisfactory": False}
<Example7>

<Example8: Ask for more information "bring an 18-inch sample">
Expert's Answer:
Thank you for your question. For the best identification of your problem, bring an 18-inch sample (1/2 diseased & 1/2 healthy) of your tree for our tree team to accurately identify. Your pictures possibly depict fire blight, which is running rampant in the area.
Judgement:
{"satisfactory": False}
<Example8>

<Example9: Satisfactory answer>
Expert's Answer:
Greetings, This fungus is Phaeolus schweinitzii, the "dyers polypore," so-called because it is used as a dyestuff by crafters.  It is not edible.
Judgement:
{"satisfactory": True}
<Example5>

Please judge the following answer. 
"""

        # Prompt for Reformatted VQA dataset
#         prefix = """\
# I am cleaning an agricultural Q&A dataset and need your help to determine \
# if the expert's answer is satisfactory. If the expert's answer is unsatisfactory, \
# such as suggesting to contact someone else, asking for more information, \
# indicating uncertainty, or admitting they cannot help, output {"satisfactory": False}. Otherwise, output {"satisfactory": True}. 
# Here are some examples to help you understand the task better:

# <Example1: suggesting to contact someone else>
# Expert's Answer:
# Not enough information to determine what's wrong with your tree. Part of the damage looks new, yet there appears to be rough irregular damaged bark already present. Are there other symptoms of concern? I suggest you contact a professional tree expert to take a closer look at the entire tree. Here is a link for more information:\u00a0<Link 1>
# Judgement: 
# {"satisfactory": False}
# <Example1>

# <Example2: suggesting to contact someone else>
# Expert's Answer:
# Hi - I really can't tell what is wrong just from the picture. If the plant is wilting, it might be a root rot disease. We did get a lot of rain last year and through the winter. If it is a root rot disease, there is not much you can do except dig the plant out and replace it with a different species. If you want to confirm that this is the problem, you can send a sample to the UConn Plant Diagnostic Lab (<Link 1>) where they can culture your plant for diseases. Contact them first to see how you would submit a sample.
# Judgement:
# {"satisfactory": False}
# <Example2>

# <Example3: asking for more information>
# Expert's Answer: 
# Unfortunately I am not able to identify from the picture provided. Provide a better quality picture and I will give the id another attempt. Where was the plant purchased? Go to the store where purchased and see if they still have this plant? Below are a couple of links to help in identification: <Link 1> <Link 2>
# Judgement: 
# {"satisfactory": False}
# <Example3>

# <Example4: indicating uncertainty>
# Expert's Answer:
# We can not tell for sure from your photos what your issue is. Possibilities include wildlife burrowing and stormwater issues.
# Judgement: 
# {"satisfactory": False}
# <Example4>

# <Example5: admitting they cannot help>
# Expert's Answer:
# Unfortunately, our group cannot answer your question because it doesn't serve your location. Please contact your local Cooperative Extension office for assistance. A good way to find your local office is to go to <Link 1> and enter your county or parish name along with your state name. You might also use your favorite search engine and enter \"cooperative extension\" along with your county name.
# Judgement: 
# {"satisfactory": False}
# <Example5>

# <Example6: Satisfactory answer>
# Expert's Answer:
# Hello, this is a common problem in Japanese plums, which typically require more pruning than European plums. Using heading cuts when pruning will help with this problem. Below are some publications that discuss pruning plums:<Link 1>
# Judgement:
# {"satisfactory": True}
# <Example6>

# Please judge the following answer. 
# """

        sample_prompt = f"Expert's Answer:\n{item['answer']}\nJudgement:\n"
        return {"prompt": prefix + "\n" + sample_prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            response = client.chat(prompt=prompt["prompt"], response_format=Satisfactory)
            item["satisfactory"] = response.satisfactory
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["satisfactory"] = None
            
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
                        if "satisfactory" in item and item["satisfactory"] is not None:
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
                    if "satisfactory" in item and item["satisfactory"] is not None:
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

    checker = CheckSatisfactory(args.input_file, args.output_file, args.model_name, args.num_processes)
    checker.extract()
