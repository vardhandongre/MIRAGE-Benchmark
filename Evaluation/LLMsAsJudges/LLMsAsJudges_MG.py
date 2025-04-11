import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from chat_models.Gemini import Gemini
from chat_models.Claude import Claude  # Add Claude import
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
import re

class Score(BaseModel):
    accuracy: int
    relevance: int
    completeness: int
    parsimony: int

    def to_json(self):
        return {
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "parsimony": self.parsimony
        }

class Scorer:
    def __init__(self, raw_data_file, output_file, expert_name, subject_name, model_name="gpt-4o", num_processes=None, temperature=1):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.expert_name = expert_name
        self.subject_name = subject_name
        self.temperature = temperature

    def get_prompt(self, item):
        title = item["title"]
        user_query = item["question"]
        if self.expert_name not in item:
            if self.expert_name in item["b_type"]:
                expert_answer = item["b_type"][self.expert_name]
            else:
                raise ValueError(f"Expert answer not found in item {item.get('id', 'unknown')}")
        else:
            expert_answer = item[self.expert_name]
        images = item.get("attachments", [])
        if self.subject_name not in item:
            if self.subject_name in item["a_type"]:
                model_response = item["a_type"][self.subject_name]
            elif self.subject_name in item["b_type"]:
                model_response = item["b_type"][self.subject_name]
            else:
                raise ValueError(f"Model response not found in item {item.get('id', 'unknown')}")
        else:
            model_response = item[self.subject_name]
        if not model_response:
            print(f"Model response is empty for item {item.get('id', 'unknown')}")

        score_criteria = """**Accuracy Definition**: Whether the agricultural facts, species names, and diagnostic conclusions stated in the answer align with expert responses and scientific consensus. Emphasis is placed on the correctness of professional terminology (e.g., naming of disease types), accuracy of key details (e.g., descriptions of lesion characteristics), and logical coherence of causal relationships (e.g., transmission pathways of pests/diseases). *Example: Correctly identifying "wheat stripe rust" and describing "yellow linear lesions with powdery spore masses" would earn high marks, whereas confusing it with "leaf rust" or misdescribing lesion color would result in deductions.*

**Relevance Definition**: Whether the answer directly addresses the visual content (e.g., plant parts in uploaded images) and core needs of the user (e.g., emergency pest/disease management), while excluding irrelevant content. *Example: For a question about "the cause of tomato leaf curling," extensive discussion of fruit storage methods would be deemed irrelevant.*

**Completeness Definition**: Whether the model’s answer covers all key information points mentioned in expert answers—such as disease identification, preventive measures, treatment plans—to fully address the user’s inquiry. If the model omits critical steps or precautions highlighted in expert answers, it is deemed incomplete. *Example: If an expert answer outlines three steps (identifying the disease, applying chemical treatments, and post-treatment management), but the model only discusses disease identification without mentioning treatment, the response lacks completeness.* 

**Parsimony Definition:**: Whether the answer provides actionable guidance that directly addresses the user’s core needs, delivering a concise and unambiguous conclusion and specific recommendations without extraneous technical details. The response should adhere to Occam’s Razor by avoiding unnecessary complexity and focusing only on what is essential for understanding whether intervention is necessary and what exact steps (if any) need to be taken.  *Example: A response that clearly states "The tree is healthy and requires no treatment," along with a brief, direct explanation, demonstrates parsimony, whereas an extended discussion on multiple disease possibilities that are not supported by the visible evidence would be less parsimonious.*

---

### Scoring Guidelines

**1. Accuracy (0-4 points)**  
- **4 points:** All agricultural facts, terms, and causal relationships are fully correct and align with expert consensus.  
- **3 points:** Minor errors in terminology or details, but core conclusions remain accurate.  
- **2 points:** Significant factual errors or misidentified species/diseases, but partial correctness is present.  
- **1 point:** Major inaccuracies, e.g., confusing diseases or flawed causal logic.  
- **0 points:** Entirely incorrect or unscientific claims.

**2. Relevance (0-4 points)**  
- **4 points:** Directly addresses the user’s query; no irrelevant content.  
- **3 points:** Mostly relevant but includes minor tangential details.  
- **2 points:** Partially relevant but omits key user-requested elements.  
- **1 point:** Largely off-topic or misinterprets the core query.  
- **0 points:** Entirely unrelated to the user’s question.

**3. Completeness (0-4 points)**  
- **4 points:** Covers all key points from the gold answer (e.g., diagnosis, treatment, prevention).  
- **3 points:** Misses 1-2 minor details but addresses core aspects.  
- **2 points:** Omits a major component (e.g., treatment steps).  
- **1 point:** Only addresses a single aspect superficially.  
- **0 points:** Fails to address any key elements of the query.

**4. Parsimony (0-4 points)**

- **4 points:** The answer is succinct, clear, and directly addresses the user’s concerns. It offers straightforward, practical guidance that is fully aligned with the visible evidence without any unnecessary details. It embodies the principle of Occam’s Razor.
- **3 points:** The answer is generally concise and practical, offering useful advice. However, it may include some extraneous details or slight ambiguity that only minimally detracts from its overall clarity and directness.
- **2 points:** The answer contains relevant information but is overly theoretical or detailed. Extra technical content obscures the key actionable recommendations, making the response less concise and direct.
- **1 point:** The answer is largely indirect or abstract, with a significant amount of unnecessary information. The lack of clarity in actionable guidance leaves the user uncertain about whether any intervention is needed.
- **0 points:** The answer fails to provide practical or actionable recommendations and is cluttered with superfluous details, completely missing the concise, straightforward approach required by Occam’s Razor."""

        prompt = f"""
You are now required to rate a model's response to an agriculture-related question. \
We have a gold answer, which is Expert's Answer and based on this gold answer, \
and the user's question, you need to score the model's answer according to the following four scoring criteria.

<User Query>{user_query}</User Query>

<Gold Answer>{expert_answer}</Gold Answer>

<Model Response>{model_response}</Model Response>

<Score Criteria>{score_criteria}</Score Criteria>

Please only output the scores without any other content. You should output JSON with four key, accuracy, relevance, completeness, parsimony. The example is shown below:
{{ "accuracy": ..., "relevance": ..., "completeness": ..., "parsimony": ... }}"""

        system_prompt = "You are a helpful assistant that evaluates and scores responses in agricultural contexts."
        
        return {
            "prompt": prompt,
            "system": system_prompt,
            "images": images,
            "expert_answer": expert_answer,
            "model_response": model_response
        }

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)
        max_retries = 5
        retries = 0
        while retries < max_retries:
            if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
                client = GPT4O(model_name=model_name, messages=[])
            elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash" or self.model_name == "gemini-2.0-flash":
                client = Gemini(model_name=model_name, messages=[])
            elif self.model_name == "claude-3-5-sonnet-latest" or self.model_name == "claude-3-7-sonnet-latest":
                client = Claude(model_name=model_name, messages=[])
            else:
                raise ValueError(f"Model '{self.model_name}' not supported.")
            
            new_item = {
                "jugde": self.model_name,
                "id": item.get('id'),
                "title": item.get('title'),
                "question": item.get('question'),
                "expert_name": self.expert_name,
                "subject_name": self.subject_name,
                "expert_answer": prompt["expert_answer"],
                "model_response": prompt["model_response"],
                "category": item.get('category'),
            }
            
            try:
                if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
                    response = client.chat(prompt=prompt["prompt"], response_format=Score, temperature=0)
                    new_item["score"] = response.to_json()
                    response = response.to_json()
                elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash" or self.model_name == "gemini-2.0-flash":
                    response = client.chat(prompt=prompt["prompt"], response_format=Score, temperature=0)
                    response = self.extract_json(response)
                    new_item["score"] = response
                elif self.model_name == "claude-3-5-sonnet-latest" or self.model_name == "claude-3-7-sonnet-latest":
                    client.system = prompt["system"]
                    response = client.chat(prompt=prompt["prompt"])
                    response = self.extract_json(response)
                    new_item["score"] = response
                
                assert "accuracy" in response and "relevance" in response and "completeness" in response and "parsimony" in response
                new_item["info"] = client.info()
                new_item["history"] = client.get_history()
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error processing item {item.get('id', 'unknown')}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                new_item["score"] = -1
                
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        
        return new_item.get('id')

    def extract_json(self, string):
        try:
            json_data = json.loads(string)
            return json_data
        except json.JSONDecodeError:
            if isinstance(string, dict):
                return string
            
            pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
            match = re.search(pattern, string)

            if match:
                json_str = match.group(1)
            else:
                # Look for JSON object in the string
                start = string.find('{')
                end = string.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = string[start:end]
                else:
                    json_str = string.strip()

            json_data = json.loads(json_str)   
            return json_data   

    def scoring(self):
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
                        if 'score' in item and item['score'] != -1 and item['score'] != None:
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
                    if 'score' in item and item['score'] != -1 and item['score'] != None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--expert_name", type=str, required=True, help="Name of the expert's answer field.")
    parser.add_argument("--subject_name", type=str, required=True, help="Name of the subject's answer field.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model response generation.")
    
    args = parser.parse_args()
    scorer = Scorer(args.input_file, args.output_file, args.expert_name, args.subject_name, args.model_name, args.num_processes, args.temperature)
    scorer.scoring()
