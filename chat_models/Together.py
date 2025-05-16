from together import Together
import base64
import copy
import math
from PIL import Image
import os

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class llama4():
    def __init__(self, model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct", messages=[]):
        self.client = Together(api_key=TOGETHER_API_KEY)
        self.model_name = model_name
        self.messages = messages
        self.history_info = copy.deepcopy(messages)

    def chat(self, prompt, images=[], response_format=None, temperature=1):
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
                messages=self.messages,
                temperature=temperature
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            self.history_info.append({"role": "assistant", "content": response})
        else:
            # JSON format
            completion = self.client.beta.chat.completions.parse(                
                model=self.model_name,
                messages=self.messages,
                response_format=response_format,
                temperature=temperature
            )
            response = completion.choices[0].message.parsed
            self.messages.append({"role": "assistant", "content": str(response.to_json())})
            self.history_info.append({"role": "assistant", "content": str(response.to_json())})
            
        return response
 
    def info(self):
        return "Together API"
    
    def get_history(self):
        return self.history_info
