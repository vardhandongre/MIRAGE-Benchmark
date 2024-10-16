#%%
from openai import OpenAI

class GPT4O():
    def __init__(self, model_name="gpt-4o", messages=[]):
        self.client = OpenAI()
        self.model_name = model_name
        self.messages = messages
    
    def chat(self, prompt, images=[]):
        if images == []:
            self.messages.append({"role": "user", "content": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": image}})
            self.messages.append({"role": "user", "content": content})
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )
        response = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def history(self):
        return self.messages

#%%
client = GPT4O()
response = client.chat("What is the capital of France?")
response = client.chat("Please describe the image", ["https://upload.wikimedia.org/wikipedia/commons/f/f9/Kim_Ji-soo_in_Sydney_on_June_15th%2C_2019_04.png"])
print(client.history())
