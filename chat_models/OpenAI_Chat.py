from openai import OpenAI
import base64
import copy

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class GPT4O():
    def __init__(self, model_name="gpt-4o", messages=[]):
        self.client = OpenAI()
        self.model_name = model_name
        self.messages = messages
        self.history_info = copy.deepcopy(messages)
        # Define pricing per model
        self.pricing = {
            "gpt-4o": {
                "input": 2.50 / 1_000_000,  # $2.50 per 1M input tokens
                "output": 10.00 / 1_000_000,  # $10.00 per 1M output tokens
            },
            "gpt-4o-mini": {
                "input": 0.150 / 1_000_000,  # $0.150 per 1M input tokens
                "output": 0.600 / 1_000_000,  # $0.600 per 1M output tokens
            },
            "gpt-3.5-turbo-0125": {
                "input": 0.50 / 1_000_000,  # $0.50 per 1M tokens
                "output": 1.50 / 1_000_000,  # $1.50 per 1M tokens
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def chat(self, prompt, images=[], response_format=None):
        # Images
        if images == []:
            self.messages.append({"role": "user", "content": prompt})
            self.history_info.append({"role": "user", "content": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                base64_image = encode_image(image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            self.messages.append({"role": "user", "content": content})
            self.history_info.append({"role": "user", "content": prompt})
    
        if response_format is None:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            self.history_info.append({"role": "assistant", "content": response})
        else:
            # JSON format
            completion = self.client.beta.chat.completions.parse(                
                model=self.model_name,
                messages=self.messages,
                response_format=response_format
            )
            response = completion.choices[0].message.parsed
            self.messages.append({"role": "assistant", "content": str(response.to_json())})
            self.history_info.append({"role": "assistant", "content": str(response.to_json())})
            
        self.prompt_tokens = completion.usage.prompt_tokens
        self.completion_tokens = completion.usage.completion_tokens
        return response

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
        return self.history_info
