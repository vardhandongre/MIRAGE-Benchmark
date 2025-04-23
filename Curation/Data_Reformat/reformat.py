import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from chat_models.Gemini import Gemini
from chat_models.Claude import Claude
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

class ReformattedAnswer(BaseModel):
    reformatted_answer: str

    def to_json(self):
        return {
            "reformatted_answer": self.reformatted_answer
        }

class Reformat:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.output_name = model_name + "-ra"
        # If the number of processes is not specified, use the number of CPU cores
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):
#         system = """\
# I'm currently creating a VQA dataset related to agriculture. \
# Each Q&A pair features a user-provided question and images, and an answer from an expert. \
# The expert's response includes a URL. I will provide the text information from the URL. \
# You need to combine the content from the URL and the expert's answer to reorganize the expert's \
# response so that it only contains text and is high-quality, providing a more detailed answer to the user's question. \
# Please do not output links!

# The user's question by <User>, the expert's response by <Expert>, \
# and the content of the link by <Link i> (indicating the content of the first link).

# Please only output the reformatted answer in English. Please do not output links! You should output JSON with the key reformatted_answer. The eample is shown below:
# {
#     "reformatted_answer ": ...
# }"""

#         system = """\
# You are given a user’s question about an agricultural scenario, \
# an expert’s original answer (which contains one or more URLs), and the full text content from each URL. \
# Your job is to merge the expert’s answer with the URL content into a single, high‑quality, text‑only response in English. \
# Please do not output links!

# Your reformatted answer must satisfy these four criteria:

# 1. **Accuracy**  
#    - Align precisely with the expert’s original guidance and the provided URL content.  
#    - Use correct professional terminology (e.g., precise disease or pest names).  
#    - Preserve all key factual details (e.g., lesion characteristics, pest behavior).  
#    - Maintain logical coherence in describing causal relationships (e.g., transmission pathways).  
#    - Ensure every management recommendation or intervention is appropriate and effective.

# 2. **Relevance**  
#    - Stay strictly on topic: focus only on information directly tied to the user’s question and the expert’s guidance.  
#    - Avoid introducing unrelated agricultural details or digressions.

# 3. **Completeness**  
#    - Cover every critical point originally mentioned by the expert or in the URL content:  
#      - Professional terminology  
#      - Detailed descriptions of symptoms or behaviors  
#      - Explanations of causal relationships  
#      - Full set of management strategies and precautions

# 4. **Parsimony**  
#    - Provide concise, actionable guidance:  
#      - Omit any extraneous technical detail that does not aid the user’s immediate decision or understanding.  
#      - Deliver a clear conclusion and specific next steps, avoiding unnecessary complexity.

# **Input Format**  
# - `<User>`: the user’s question.  
# - `<Expert>`: the expert’s original answer (contains URLs).  
# - `<Link1>`, `<Link2>`, etc.: the textual content extracted from each URL.

# **Output Format**  (Please do not output links!)
# {
#   "reformatted_answer": "Your combined, text‑only answer here."
# }
# """     
        system = """\
You are given a user’s question about an agricultural scenario, an expert’s original answer (which contains one or more URLs), and the full text content from each URL. Your job is to merge the expert’s answer with the URL content into a single, high‑quality, text‑only response in English—without any links—formatted as JSON under the key `reformatted_answer`.

Your reformatted answer must use **only** information present in the expert’s original answer and the provided URL content. Do not introduce any external facts or omit any details from those sources.

Your reformatted answer must satisfy these four criteria:

1. **Accuracy**  
   - Align precisely with the expert’s original guidance and the provided URL content.  
   - Use correct professional terminology (e.g., precise disease or pest names).  
   - Preserve all key factual details (e.g., lesion characteristics, pest behavior).  
   - Maintain logical coherence in describing causal relationships (e.g., transmission pathways).  
   - Ensure every management recommendation or intervention is appropriate and effective.

2. **Relevance**  
   - Stay strictly on topic: focus only on information directly tied to the user’s question and the expert’s guidance.  
   - Avoid introducing unrelated agricultural details or digressions.

3. **Completeness**  
   - Cover every critical point originally mentioned by the expert or in the URL content:  
     - Professional terminology  
     - Detailed descriptions of symptoms or behaviors  
     - Explanations of causal relationships  
     - Full set of management strategies and precautions

4. **Parsimony**  
   - Provide concise, actionable guidance:  
     - Omit any extraneous technical detail that does not aid the user’s immediate decision or understanding.  
     - Deliver a clear conclusion and specific next steps, avoiding unnecessary complexity.

**Input Format**  
- `<User>`: the user’s question.  
- `<Expert>`: the expert’s original answer (contains URLs).  
- `<Link1>`, `<Link2>`, etc.: the textual content extracted from each URL.

**Output Format**  
{
  "reformatted_answer": "Your combined, text‑only answer here."
}
"""

        question = item["question"]
        answer = item["answer"]
        images = item.get("attachments", [])
        urls = item.get("urls", [])
        content = ""
        url_contents = {}
        for url in urls:
            url_id = url['url_id'].split('_')[-1]
            url_title = url['title']
            url_content = url['content']
            url_contents[url_id] = f"Title: {url_title}\n\nContent:{url_content}"

        for i in range(1, len(urls)+1):
            content += f"<Link {i}>" + '\n' + url_contents[str(i)] + '\n' + f"</Link {i}>" + '\n'
        
        user = f"<User>{question}</User>\n<Expert>{answer}</Expert>\n{content}"
        
        return {"system": system, "user": user, "images": images}

    # Function to handle item processing
    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name.startswith("gpt"):
            client = GPT4O(model_name=model_name, messages=[{"role": "system", "content": prompt["system"]}])
        elif self.model_name.startswith("gemini"):
            client = Gemini(model_name=model_name, messages=[prompt["system"]])
        elif self.model_name == "claude-3-5-sonnet-latest":
            system_prompt = prompt["system"] + "\nVery important: Your response must be a valid JSON string with exactly this format: {\"reformatted_answer\": \"your detailed answer here\"}"
            client = Claude(model_name=model_name, messages=[])
            client.system = system_prompt
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            response = client.chat(prompt=prompt["user"], images=prompt["images"], response_format=ReformattedAnswer)
            if self.model_name.startswith("gpt"):
                item[self.output_name] = response.reformatted_answer
            elif self.model_name.startswith("gemini"):
                response = json.loads(response)
                item[self.output_name] = response["reformatted_answer"]
            elif self.model_name == "claude-3-5-sonnet-latest":
                response = json.loads(response)
                item[self.output_name] = response["reformatted_answer"]
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            # Handle errors gracefully and log them
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item[self.output_name] = -1
            
        # Lock the file access to avoid race conditions
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                # Each item is written as a single JSON line
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def reformat(self):
        # Read the raw data file
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Check if the output file exists and read processed items
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if self.output_name in item and item[self.output_name] != -1 and item[self.output_name] != None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        # Handle potentially corrupt JSON lines
                        continue
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            # Initialize the process pool with the specified number of processes
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock) for item in items_to_process]
            # Use tqdm to show progress
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
                    if self.output_name in item and item[self.output_name] != -1 and item[self.output_name] != None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    reformatter = Reformat(raw_data_file=args.input_file, output_file=args.output_file, model_name=args.model_name, num_processes=args.num_processes)
    reformatter.reformat()
