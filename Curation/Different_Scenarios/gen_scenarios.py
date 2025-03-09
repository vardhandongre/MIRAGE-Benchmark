import sys
sys.path.append('../')
import json
import multiprocessing
import os
import random
from pydantic import BaseModel
from tqdm import tqdm
from chat_models import GPT4O, Gemini

class QA(BaseModel):
    question: str
    answer: str
    
    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer
        }

class QAGenerator:
    def __init__(self, input_file, output_file, model_name="gpt-4o-mini", scenario_weights=None, num_processes=None):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.scenario_weights = scenario_weights if scenario_weights is not None else {"standard": 1, "farm_manager_and_agricultural_consultant": 1, "ai_model_assist_farmer": 1, "evaluator_and_ai_model": 1}
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

        standard_system_prompt = """You need to generate a question-and-answer pair based on these agriculture images and image description. \
        The question should be designed to test other models’ understanding of these agriculture images; it should be phrased simply and \
        conversationally. However, your response should be professional, showcasing your understanding \
        of the images by providing useful information derived from the image and detailed analysis. \
        The reply should offer detailed and rich useful information."""

        # Farm Manager and Agricultural Consultant
        farm_manager_and_agricultureal_consultant = """You need to generate a question-and-answer pair based on these agriculture images and image description.\
        You should take on the roles of a farm manager and an agricultural consultant, discussing the conditions shown in the image. \
        The consultant should explain the visible agricultural conditions in simple terms and answer any questions posed by the farm manager. \
        The manager might inquire about crop health, potential issues, soil quality, or pest presence. \
        The consultant should provide clear, actionable advice to ensure the manager understands the situation and can make informed decisions, \
        showcasing your understanding of the images by providing useful information derived from the image and detailed analysis.
        """

        # Agricultural Consultant and Agricultural Consultant
        agricultural_consultant_and_agricultural_consultant = """You need to generate a question-and-answer pair based on these agriculture images and image description. \
        This pair should be a professional discussion between two agricultural consultants analyzing the images. \
        You need to mimic an expert tone in asking and answering questions. The conversation should focus on detailed \
        aspects derived from the images, such as crop health indicators, pest infestations, soil conditions, and potential yield outcomes. \
        The response should include technical insights and considerations for improving or maintaining optimal agricultural practices.""" 


        # AI Model Assisting Farmer
        ai_model_assist_farmer = """You need to generate a question-and-answer pair based on these agriculture images and image description. \
        You need to act as an AI model assisting a farmer who has questions about visible content on their agricultural images. \
        The farmer may be curious or concerned about crop conditions, soil health, or pest infestations. \
        The AI model should explain specific details such as plant vigor, soil moisture levels, or signs of disease, maintaining simplicity and avoiding excessive technical jargon. \
        The AI model’s response should aim to provide educational insights to help the farmer better understand their fields, showcasing your understanding of the images by providing useful information derived from the image and detailed analysis.
        """

        # Evaluator and AI Model
        evaluator_and_ai_model = """You need to generate a question-and-answer pair based on these agriculture images and image description. \
        Act as a member of a quality control team, focusing on assessing an AI model’s visual capabilities in analyzing agricultural images. \
        The evaluator should inquire about subtle details in the images, such as signs of stress in plants, nutrient deficiencies, or pest damage, \
        to ensure the AI model can accurately identify and interpret these aspects."""

        # Agricultural Educator and Student
        agricultural_educator_and_student = """You need to generate a question-and-answer pair based on these agriculture images and image description. \
        In this context, an agricultural educator is explaining the images to a student. The student should ask questions to clarify their understanding \
        of basic agricultural concepts such as crop growth stages, soil types, or pest management strategies. The educator should offer simplified explanations, \
        using the images to illustrate key points and teaching the student how to identify important agricultural indicators."""

        self.scenarios = {
            "standard": standard_system_prompt,
            "farm_manager_and_agricultural_consultant": farm_manager_and_agricultureal_consultant,
            "agricultural_consultant_and_agricultural_consultant": agricultural_consultant_and_agricultural_consultant,
            "ai_model_assist_farmer": ai_model_assist_farmer,
            "evaluator_and_ai_model": evaluator_and_ai_model,
            "agricultural_educator_and_student": agricultural_educator_and_student
        }

        self.suffix = "The description of the image is marked by <Des>. YOU SHOULD OUTPUT JSON FORMAT."

    def choose_scenario(self):
        # Randomly select a scenario based on the provided weights
        scenarios = list(self.scenario_weights.keys())
        weights = list(self.scenario_weights.values())
        return random.choices(scenarios, weights=weights, k=1)[0]

    def get_prompt(self, sample, scenario):
        system_prompt = self.scenarios[scenario] + "\n" + self.suffix
        return system_prompt, f"<Des> {sample['description']}<\Des>"

    def process_item(self, args):
        sample, model_name, output_file, lock = args
        scenario = self.choose_scenario()
        system_message, prompt = self.get_prompt(sample, scenario)
        
        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[{"role": "system", "content": system_message}])
        elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash":
            client = Gemini(model_name=model_name, messages=[system_message])

        try:
            response = client.chat(prompt=prompt, images=sample['attachments'], response_format=QA)
            if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
                response_data = response.to_json()
            elif self.model_name == "gemini-1.5-pro" or self.model_name == "gemini-1.5-flash":
                response_data = json.loads(response)            
            sample["systhetic_qa"] = {"scenario": scenario, "question": response_data["question"], "answer": response_data["answer"]}
            sample["info"] = client.info()
            sample["history"] = client.get_history()

        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            sample["systhetic_qa"] = -1

        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        return sample.get('id')

    def generate(self):
        try:
            with open(self.input_file, "r", encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
            with open(self.input_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "systhetic_qa" in item and item["systhetic_qa"] != -1:
                            processed_ids.add(item.get('id'))
                    except json.JSONDecodeError:
                        continue

        items_to_process = [sample for sample in data if sample.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")

        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(sample, self.model_name, self.output_file, lock) for sample in items_to_process]
            for _ in tqdm(pool.imap_unordered(self.process_item, args_list), total=len(args_list), desc="Generating Q&A"):
                pass
            pool.close()
            pool.join()

        print("Generation completed.")
        self.cleanup_output(len(data))

    def cleanup_output(self, data_length):
        valid_items = []

        with open(self.output_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "systhetic_qa" in item and item["systhetic_qa"] != -1:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Q&A pairs using scenarios and weights.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--scenario_weights", type=json.loads, default='{"standard": 1, "farm_manager_and_agricultural_consultant": 1, "ai_model_assist_farmer": 1, "evaluator_and_ai_model": 1}', help="JSON string representing scenario weights.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()
    
    print(f"scenario_weights: {args.scenario_weights}")
    generator = QAGenerator(args.input_file, args.output_file, args.model_name, args.scenario_weights, args.num_processes)
    generator.generate()
