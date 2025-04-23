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
class Satisfactory(BaseModel):
    analysis: str
    unsatisfactory: bool
    
    def to_json(self):
        return {"analysis": self.analysis, "unsatisfactory": self.unsatisfactory}

class CheckSatisfactory:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Modify the prompt to check if the expert's answer is unsatisfactory with analysis
    def get_prompt(self, item):
    # B1 Type
#         prefix = """\
# I am cleaning an agricultural Q&A dataset and need your help to determine if the expert's answer is unsatisfactory. Please analyze the expert's answer first, then provide your judgment. \
# Your output should be in JSON format like this: {"analysis": "...", "unsatisfactory": true/false}.

# An expert's answer is UNSATISFACTORY if it:
# 1. Suggests contacting someone else (another expert, extension office, professional, etc.)
# 2. Asks for more information (e.g., "send another picture", "provide more details", "provide a sample:", expert's response exist “?”)
# 3. Expresses uncertainty (e.g., "can't tell for sure", "not able to identify", "possibilities include")
# 4. Admits inability to help (e.g., "cannot answer your question", "our group doesn't serve your location")
# 5. Expert doesn't answer the question, and asks user to search online for the answer.
# 6. Recommending experts

# An expert's answer is SATISFACTORY ONLY if it:
# 1. Directly addresses the question
# 2. Provides complete information without requiring external resources
# 3. Shows confidence in the expertise being provided
# 4. Doesn't defer to other experts or resources for the main answer

# Examples:

# <Example1>
# Expert's Answer:
# Not enough information to determine what's wrong with your tree. Part of the damage looks new, yet there appears to be rough irregular damaged bark already present. Are there other symptoms of concern? I suggest you contact a professional tree expert to take a closer look at the entire tree. Here is a link for more information: <Link 1>

# Model output:
# {
# "analysis": "Expert suggests contacting someone else, and asks for more information, expert asks "Are there other symptoms of concern?",
# "unsatisfactory": True
# }
# </Example1>

# <Example2>
# Expert's Answer:
# Hi - I really can't tell what is wrong just from the picture. If the plant is wilting, it might be a root rot disease. We did get a lot of rain last year and through the winter. If it is a root rot disease, there is not much you can do except dig the plant out and replace it with a different species. If you want to confirm that this is the problem, you can send a sample to the UConn Plant Diagnostic Lab (<Link 1>) where they can culture your plant for diseases. Contact them first to see how you would submit a sample.

# Model output:
# {
# "analysis": "Expert expresses uncertainty ("can't tell"), refers to a diagnostic lab, and suggests contacting them.",
# "unsatisfactory": True
# }
# </Example2>

# <Example3>
# Expert's Answer: 
# Unfortunately I am not able to identify from the picture provided. Provide a better quality picture and I will give the id another attempt. Where was the plant purchased? Go to the store where purchased and see if they still have this plant? Below are a couple of links to help in identification: <Link 1> <Link 2>

# Model output:
# {
# "analysis": "Expert admits inability to identify, asks for more information (better picture).",
# "unsatisfactory": True
# }
# </Example3>

# <Example4>
# Expert's Answer:
# We can not tell for sure from your photos what your issue is. Possibilities include wildlife burrowing and stormwater issues.

# Model output:
# {
# "analysis": "Expert indicates uncertainty ("cannot tell for sure") and only offers possibilities rather than a definitive answer.",
# "unsatisfactory": True
# }
# </Example4>

# <Example5>
# Expert's Answer:
# Unfortunately, our group cannot answer your question because it doesn't serve your location. Please contact your local Cooperative Extension office for assistance. A good way to find your local office is to go to <Link 1> and enter your county or parish name along with your state name. You might also use your favorite search engine and enter \"cooperative extension\" along with your county name.

# Model output:
# {
# "analysis": "Expert admits inability to help and directs to contact someone else.",
# "unsatisfactory": True
# }
# </Example5>

# <Example6>
# Expert's Answer:
# This is fire blight, a bacterial disease that affects plants in the rose family, particularly pear and apple trees. The characteristic shepherd's crook at the end of branches and blackened leaves are distinctive symptoms. Remove affected branches by cutting at least 12 inches below visible infection. Disinfect your pruning tools between each cut using a 10% bleach solution. Avoid excessive nitrogen fertilization which promotes susceptible new growth. Some copper-based fungicides can help prevent spread but won't cure existing infections.

# Model output:
# {
# "analysis": "Expert provides a clear identification, explains the symptoms, and offers specific actionable advice without referring elsewhere.",
# "unsatisfactory": False
# }
# </Example6>

# <Example7>
# Expert's Answer:
# This appears to be powdery mildew on your zucchini plants. It's a fungal disease that thrives in humid conditions with poor air circulation. To manage it: 1) Remove severely infected leaves, 2) Avoid overhead watering and water at the base of plants in the morning, 3) Ensure adequate spacing between plants for better airflow, 4) Apply a fungicide labeled for powdery mildew on cucurbits - options include sulfur-based products or potassium bicarbonate. Prevention includes choosing resistant varieties for next season.

# Model output:
# {
# "analysis": "Expert confidently identifies the problem and provides complete, actionable advice without referring to external resources.",
# "unsatisfactory": False
# }
# </Example7>

# <Example8>
# Expert's Answer:
# I'm guessing it's a spider. You can search for spiders and compare them online to see which one matches yours best.

# Model output:
# {
# "analysis": "Expert shows the uncertainty and ask user to search information online.",
# "unsatisfactory": True
# }
# <Example8>

# <Example9>
# Expert's Answer:
# The local insect expert is Kevin. His email is <personal data hidden>.

# Model output:
# {
# "analysis": "Expert recommends local experts.",
# "unsatisfactory": True
# }
# <Example9>

# <Example10>
# Expert's Answer:
# What you saw was a turbulent phosphila caterpillar that turns into an ugly moth. It feeds on greenbrier. If you google the the name, you will find all the information on this caterpillar. Thank you for your question.

# Model output:
# {
# "analysis": "Although expert asks user to search online, but expert already answered the question.",
# "unsatisfactory": False
# }
# <Example10>

# Please analyze and judge the following answer:"""

        # B2 Type
        prefix = """\
I am cleaning an agricultural Q&A dataset and need your help to determine if the expert's answer is unsatisfactory. Please analyze the expert's answer first, then provide your judgment. \
Your output should be in JSON format like this: {"analysis": "...", "unsatisfactory": true/false}.

An expert's answer is UNSATISFACTORY if it:
1. Suggests contacting someone else (another expert, extension office, professional, etc.)
2. Asks for more information (e.g., "send another picture", "provide more details", "provide a sample:", expert's response exist “?”)
3. Expresses uncertainty (e.g., "can't tell for sure", "not able to identify", "possibilities include")
4. Admits inability to help (e.g., "cannot answer your question", "our group doesn't serve your location")
5. Expert doesn't answer the question, and asks user to search online for the answer.
6. Recommending experts

An expert's answer is SATISFACTORY ONLY if it:
1. Directly addresses the question
2. Shows confidence in the expertise being provided

Examples:

<Example1>
Expert's Answer:
Not enough information to determine what's wrong with your tree. Part of the damage looks new, yet there appears to be rough irregular damaged bark already present. Are there other symptoms of concern? I suggest you contact a professional tree expert to take a closer look at the entire tree. Here is a link for more information: <Link 1>

Model output:
{
"analysis": "Expert suggests contacting someone else, and asks for more information, expert asks "Are there other symptoms of concern?",
"unsatisfactory": True
}
</Example1>

<Example2>
Expert's Answer:
Hi - I really can't tell what is wrong just from the picture. If the plant is wilting, it might be a root rot disease. We did get a lot of rain last year and through the winter. If it is a root rot disease, there is not much you can do except dig the plant out and replace it with a different species. If you want to confirm that this is the problem, you can send a sample to the UConn Plant Diagnostic Lab (<Link 1>) where they can culture your plant for diseases. Contact them first to see how you would submit a sample.

Model output:
{
"analysis": "Expert expresses uncertainty ("can't tell"), refers to a diagnostic lab, and suggests contacting them.",
"unsatisfactory": True
}
</Example2>

<Example3>
Expert's Answer: 
Unfortunately I am not able to identify from the picture provided. Provide a better quality picture and I will give the id another attempt. Where was the plant purchased? Go to the store where purchased and see if they still have this plant? Below are a couple of links to help in identification: <Link 1> <Link 2>

Model output:
{
"analysis": "Expert admits inability to identify, asks for more information (better picture).",
"unsatisfactory": True
}
</Example3>

<Example4>
Expert's Answer:
We can not tell for sure from your photos what your issue is. Possibilities include wildlife burrowing and stormwater issues.

Model output:
{
"analysis": "Expert indicates uncertainty ("cannot tell for sure") and only offers possibilities rather than a definitive answer.",
"unsatisfactory": True
}
</Example4>

<Example5>
Expert's Answer:
Unfortunately, our group cannot answer your question because it doesn't serve your location. Please contact your local Cooperative Extension office for assistance. A good way to find your local office is to go to <Link 1> and enter your county or parish name along with your state name. You might also use your favorite search engine and enter \"cooperative extension\" along with your county name.

Model output:
{
"analysis": "Expert admits inability to help and directs to contact someone else.",
"unsatisfactory": True
}
</Example5>

<Example6>
Expert's Answer:
This is fire blight, a bacterial disease that affects plants in the rose family, particularly pear and apple trees. The characteristic shepherd's crook at the end of branches and blackened leaves are distinctive symptoms. Remove affected branches by cutting at least 12 inches below visible infection. Disinfect your pruning tools between each cut using a 10% bleach solution. Avoid excessive nitrogen fertilization which promotes susceptible new growth. Some copper-based fungicides can help prevent spread but won't cure existing infections.

Model output:
{
"analysis": "Expert provides a clear identification, explains the symptoms, and offers specific actionable advice without referring elsewhere.",
"unsatisfactory": False
}
</Example6>

<Example7>
Expert's Answer:
This appears to be powdery mildew on your zucchini plants. It's a fungal disease that thrives in humid conditions with poor air circulation. To manage it: 1) Remove severely infected leaves, 2) Avoid overhead watering and water at the base of plants in the morning, 3) Ensure adequate spacing between plants for better airflow, 4) Apply a fungicide labeled for powdery mildew on cucurbits - options include sulfur-based products or potassium bicarbonate. Prevention includes choosing resistant varieties for next season.

Model output:
{
"analysis": "Expert confidently identifies the problem and provides complete, actionable advice without referring to external resources.",
"unsatisfactory": False
}
</Example7>

<Example8>
Expert's Answer:
I'm guessing it's a spider. You can search for spiders and compare them online to see which one matches yours best.

Model output:
{
"analysis": "Expert shows the uncertainty and ask user to search information online.",
"unsatisfactory": True
}
<Example8>

<Example9>
Expert's Answer:
The local insect expert is Kevin. His email is <personal data hidden>.

Model output:
{
"analysis": "Expert recommends local experts.",
"unsatisfactory": True
}
<Example9>

<Example10>
Expert's Answer:
What you saw was a turbulent phosphila caterpillar that turns into an ugly moth. It feeds on greenbrier. If you google the the name, you will find all the information on this caterpillar. Thank you for your question.

Model output:
{
"analysis": "Although expert asks user to search online, but expert already answered the question.",
"unsatisfactory": False
}
<Example10>

Please analyze and judge the following answer:"""

        sample_prompt = f"Expert's Answer:\n{item['answer']}\n\nModel output:\n"

        return {"prompt": prefix + "\n" + sample_prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name.startswith("gpt"):
            client = GPT4O(model_name=model_name, messages=[])
        elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash" or self.model_name == "gemini-2.0-flash":
            client = Gemini(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            if self.model_name.startswith("gpt"):
                response = client.chat(prompt=prompt["prompt"], response_format=Satisfactory, temperature=0)
                item["unsatisfactory"] = response.unsatisfactory
                item["unsatisfactory_analysis"] = response.analysis
            elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash" or self.model_name == "gemini-2.0-flash":
                response = client.chat(prompt=prompt["prompt"], response_format=Satisfactory, temperature=0)
                response = json.loads(response)
                item["unsatisfactory"] = response["unsatisfactory"]
                item["unsatisfactory_analysis"] = response["analysis"]
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["unsatisfactory"] = None
            
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
                        if "unsatisfactory" in item and item["unsatisfactory"] is not None:
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
                    if "unsatisfactory" in item and item["unsatisfactory"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if expert answers are unsatisfactory.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    checker = CheckSatisfactory(args.input_file, args.output_file, args.model_name, args.num_processes)
    checker.extract()
