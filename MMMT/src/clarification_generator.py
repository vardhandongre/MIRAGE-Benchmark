import openai
import json
from src.config import MODEL_NAME, OPENAI_KEY

openai.api_key = OPENAI_KEY

def call_gpt(prompt):
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        seed=42
    )
    content = response['choices'][0]['message']['content'].strip()
    return extract_json(content)


import re

def extract_json(output):
    if output.startswith("```json"):
        output = output[7:].strip()
    if output.endswith("```"):
        output = output[:-3].strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{(?:[^{}]|(?R))*\}", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                raise ValueError("Malformed JSON after fallback.")
        raise ValueError("No JSON found.")