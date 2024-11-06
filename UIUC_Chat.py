import requests
import base64
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UIUC_CHAT_API = os.getenv("UIUC_CHAT_API")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class UIUC_Chat:
    def __init__(self, model_name="gpt-4o", messages=[], retrieval_only = False):
        self.url = "https://uiuc.chat/api/chat-api/chat"
        self.model_name = model_name
        self.openai_api_key = OPENAI_API_KEY
        self.api_key = "uc_f70cc027863b42a8bb56e91c64d5ba24"
        self.course_name = "cropwizard-1.5"
        self.messages = messages
        self.retrieval_only = retrieval_only

    def chat(self, prompt, images=[], temperature=0.1, stream=True):

        headers = {
            'Content-Type': 'application/json'
        }

        if images == []:
            self.messages.append({"role": "user", "content": prompt})
        else:
            content = []
            # Add images to the content if any
            for image_path in images:
                base64_image = encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

            # Add the prompt text
            content.append({
                "type": "text",
                "text": prompt
            })

            # Append the new user message to the message history
            self.messages.append({
                "role": "user",
                "content": content
            })

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "openai_key": self.openai_api_key,
            "temperature": temperature,
            "course_name": self.course_name,
            "stream": stream,
            "api_key": self.api_key,
            "retrieval_only": self.retrieval_only
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code == 200:
            # Assuming the API returns a JSON response with 'content'
            reply = response.text
            # Append the assistant's reply to the message history
            self.messages.append({
                "role": "assistant",
                "content": reply
            })
            return reply
        else:
            # Handle errors
            print(f"Error {response.status_code}: {response.text}")
            return None

    def history(self):
        return self.messages