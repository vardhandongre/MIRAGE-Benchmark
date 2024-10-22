from openai import OpenAI
import base64

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class GPT4O():
    def __init__(self, model_name="gpt-4o", messages=[]):
        self.client = OpenAI()
        self.model_name = model_name
        self.messages = messages
    
    def chat(self, prompt, images=[], response_format=None):
        # Images
        if images == []:
            self.messages.append({"role": "user", "content": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                base64_image = encode_image(image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            self.messages.append({"role": "user", "content": content})
    
        if response_format is None:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
        else:
            # JSON format
            completion = self.client.beta.chat.completions.parse(                
                model=self.model_name,
                messages=self.messages,
                response_format=response_format
            )
            response = completion.choices[0].message.parsed
            self.messages.append({"role": "assistant", "content": str(response.to_json())})
        return response
    
    def history(self):
        return self.messages
