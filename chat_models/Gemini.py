import os
from PIL import Image
import google.generativeai as genai
import time
import copy 
import json
# Configure the generative AI API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class Gemini:
    def __init__(self, model_name="gemini-1.5-pro", messages=[]):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.model_name = model_name
        self.messages = messages
        self.history = copy.deepcopy(messages)
        self.max_retries = 5
        # Define pricing per model
        self.pricing = {
            "gemini-1.5-pro": {
                "input": 1.25 / 1_000_000,  # $1.25 per 1M input tokens
                "output": 5.00 / 1_000_000,  # $5.00 per 1M output tokens
            },
            "gemini-1.5-flash": {
                "input": 0.075 / 1_000_000,  # $0.075 per 1M input tokens
                "output": 0.30 / 1_000_000,  # $0.30 per 1M output tokens
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, prompt, images=[], response_format=None):
        # Prepare message with images
        # FIXME: Message format is based on the OPENAI API, and may need to be adjusted for the GEMINI API
        message = copy.deepcopy(self.messages)
        for image_path in images:
            image = Image.open(image_path)
            message.append(image)
        
        message.append(prompt)
        self.messages.append({"role": "user", "content": message})
        self.history.append({"role": "user", "content": prompt})
        # Retry mechanism
        attempts = 0
        while attempts <= self.max_retries:
            try:
                # Determine response format
                if response_format:
                    response = self.model.generate_content(
                        message,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=response_format
                        ),
                    )
                    text_response = response.text
                else:
                    response = self.model.generate_content(message)
                    text_response = response.text
                # Assume response gives usage data for token calculations
                self.prompt_tokens = response.usage_metadata.prompt_token_count
                self.completion_tokens = response.usage_metadata.candidates_token_count

                self.messages.append({"role": "assistant", "content": text_response})
                self.history.append({"role": "assistant", "content": text_response})

                return text_response
            except Exception as e:
                print(f"Error: {e}")
                print("API limit reached. Waiting for 65 seconds before retrying...")
                time.sleep(1)
                attempts += 1      

    def info(self):
        model_name = self.model_name

        model_pricing = self.pricing.get(model_name)
        if not model_pricing:
            raise ValueError(f"Pricing information for model '{model_name}' not found.")

        input_cost = self.prompt_tokens * model_pricing["input"]
        output_cost = self.completion_tokens * model_pricing["output"]

        total_cost = input_cost + output_cost
        info = {"model": self.model_name, "cost": total_cost, "input_tokens": self.prompt_tokens, "output_tokens": self.completion_tokens}
        return info

    def get_history(self):
        return self.history
