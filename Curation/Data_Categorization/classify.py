import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse

class Category(BaseModel):
    category: str

    def to_json(self):
        return {"category": self.category}

class ClassifyVQA:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    def get_prompt(self, item):
#         system_prompt = """Please classify the given agricultural multimodal question-answer data into the following categories. Each category includes a detailed description to ensure accurate classification:

# 1. **Plant Identification**: This category includes questions focused solely on identifying the type or species of a plant, without any additional information or context.

# 2. **Insect and Pest Identification**: This includes questions and answers that focus on identifying pests, but do not offer any suggestions or methods for dealing with the pests.

# 3. **Plant Disease Identification**: This includes questions and answers that focus on identifying plant disease, but do not offer any suggestions or methods for dealing with the disease.

# 4. **Plant Health and Disease Management**: Involves providing advice on managing or improving plant health, including prevention or treatment of diseases. If the pair includes both plant disease identification and management recommendations, it belongs to this category.

# 5. **Insect and Pest Management**: The question-answer pair must include suggestions or methods for dealing with pests. If the pair includes both pest identification and management recommendations, it belongs to this category.

# 6. **Weeds/Invasive Plants Management**: The question-answer pair must include suggestions or methods for managing weeds or invasive species. If the pair includes both weed identification and management recommendations, it falls under this category.

# 7. **Plant Care and Gardening Guidance**: Provides advice on routine plant care and general gardening techniques.

# 8. **Others**: Includes any questions that do not fit into the above categories, such as non-agricultural questions or those that cannot be classified. This also includes scenarios where the expert does not directly answer the user's question, such as asking for more information, suggesting contacting someone else, or indicating they cannot help.

# Please select the most appropriate category based on the content of the question and answer."""

        # Classify based on question, answer, and title
#         system_prompt = """Please classify the given agricultural multimodal question-answer data into the following categories. Each category includes a detailed description to ensure accurate classification:

# 1. **Plant Identification**: This category includes questions focused solely on identifying the type or species of a plant, without any additional information or context.

# 2. **Insect and Pest Identification**: This includes questions and answers that focus on identifying pests, but do not offer any suggestions or methods for dealing with the pests.

# 3. **Plant Disease Identification**: This includes questions and answers that focus on identifying plant disease, but do not offer any suggestions or methods for dealing with the disease.

# 4. **Plant Disease Management**: Involves providing advice on treatment of plant diseases. If the pair includes both plant disease identification and management recommendations, it belongs to this category.

# 5. **Insect and Pest Management**: The question-answer pair must include suggestions or methods for dealing with pests. If the pair includes both pest identification and management recommendations, it belongs to this category.

# 6. **Weeds/Invasive Plants Management**: The question-answer pair must include suggestions or methods for managing weeds or invasive species. If the pair includes both weed identification and management recommendations, it falls under this category.

# 7. **Plant Care and Gardening Guidance**: Provides advice on managing or improving plant health and general gardening techniques. 

# 8. **Others**: Includes any questions that do not fit into the above categories, such as non-agricultural questions or those that cannot be classified. This also includes scenarios where the expert does not directly answer the user's question, such as asking for more information, suggesting contacting someone else, or indicating they cannot help. 

# Please select the most appropriate category based on the content of the question and answer."""


        # Classify only based on question
        system_prompt = """Please classify the given agricultural multimodal question-answer data into the following categories. Each category includes a detailed description to ensure accurate classification:

1. **Plant Identification**: This category includes questions focused solely on identifying the type or species of a plant, without any additional information or context.

2. **Insect and Pest Identification**: This includes questions that focus on identifying pests, but do not ask any suggestions or methods for dealing with the pests.

3. **Plant Disease Identification**: This includes questions that focus on identifying plant disease, but do not ask any suggestions or methods for dealing with the disease.

4. **Plant Disease Management**: Involves providing advice on treatment of plant diseases. If the question ask both plant disease identification and management recommendations, it belongs to this category.

5. **Insect and Pest Management**: The question must ask suggestions or methods for dealing with pests. If the question asks both pest identification and management recommendations, it belongs to this category.

6. **Weeds/Invasive Plants Management**: The question mustask  suggestions or methods for managing weeds or invasive species. If the question asks both weed identification and management recommendations, it falls under this category.

7. **Plant Care and Gardening Guidance**: Ask for advice on managing or improving plant health and general gardening techniques. 

8. **Others**: Includes any questions that do not fit into the above categories, such as non-agricultural questions or those that cannot be classified.

Please select the most appropriate category based on the content of the question."""

        # Categories from Yunze
#         system_prompt = """You are a helpful assistant tasked with categorizing farming-related questions by their question type. \
# The question type categories to use are: 'disease', 'weeds/invasive plants management', 'insects/pests control', 'growing advice', 'environmental stress', 'nutrient deficiency', 'generic identification', or 'other'.
# 'insect control' is for any question that is related to insect issues. 'disease' is for any question about a disease or virus. 'growing advice' is for any question about how to grow or take care of a plant. 'environmental stress' is for any questions that pertain to problems caused by the environment such as heat. 'nutrient deficiency' is for problems that are related to nutrient deficiencies like fertilizers. 'generic identification' is for questions that are purely for entity identification, with nothing related to management or other issues. Only categorize in 'other' as a LAST RESORT.
# """
        prompt = f"Title: {item['title']}\nQuestion: {item['question']}"
        
        return {"system": system_prompt, "prompt": prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[{"role": "system", "content": prompt["system"]}])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")

        try:
            response = client.chat(prompt=prompt["prompt"], response_format=Category)
            item["category"] = response.category
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["category"] = -1

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
                        if "category" in item and item["category"] != -1:
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
                    if "category" in item and item["category"] != -1:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify VQA using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    classifier = ClassifyVQA(args.input_file, args.output_file, args.model_name, args.num_processes)
    classifier.extract()
