import re
import json
import random
import argparse
from tqdm import tqdm
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

FIRST_NAMES = [
    "Alex", "Taylor", "Jordan", "Casey", "Morgan", "Drew", "Riley", "Quinn",
    "Cameron", "Avery", "Skyler", "Reese", "Peyton", "Jamie", "Elliot", "Emerson",
    "Charlie", "Logan", "Dakota", "Harper"
]

# Cached name map to keep replacements consistent
name_map = {}

def replace_names(text):
    # Matches simple names (e.g., "Thanks, James", "Dear Lesley")
    pattern = r"(Dear|Hi|Hello|Thanks|Thank you|Regards|Sincerely|Respectfully),?\s+([A-Z][a-z]+)(\s+[A-Z][a-z]+)?"

    def replacer(match):
        prefix, first, last = match.groups()
        if first not in name_map:
            name_map[first] = random.choice(FIRST_NAMES)
        return f"{prefix}, {name_map[first]}"

    return re.sub(pattern, replacer, text)

def remove_emails_and_phones(text):
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "<EMAIL>", text)
    text = re.sub(r"\(\d{3}\)\s*\d{3}-\d{4}", "<PHONE>", text)
    text = re.sub(r"\d{3}-\d{3}-\d{4}", "<PHONE>", text)
    return text

def sanitize_text(text):
    text = replace_names(text)
    text = remove_emails_and_phones(text)
    return text

def sanitize_entry(entry, use_gpt=False):
    fields_to_clean = [
        "clarification_question", "response", "gold_revealed_fact", "title"
    ]
    for field in fields_to_clean:
        if entry.get(field):
            entry[field] = sanitize_text(entry[field])

    for turn in entry["dialog_context"]:
        turn["text"] = sanitize_text(turn["text"])

    if use_gpt:
        # Use GPT-4o-mini to double check
        content = json.dumps(entry, indent=2)
        prompt = f"Sanitize the following JSON by removing or masking any personally identifiable information (PII). This includes names, emails, phone numbers, and personal addresses. Do not alter URLs unless they contain names.

```
{content}
```"
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048
        )
        try:
            raw = response['choices'][0]['message']['content']
            return json.loads(raw)
        except Exception as e:
            print(f"⚠️ GPT parsing failed: {e}")
            return entry

    return entry

def sanitize_dataset(input_path, output_path, use_gpt=False):
    with open(input_path) as f:
        data = json.load(f)

    cleaned = []
    for entry in tqdm(data, desc="Sanitizing"):
        cleaned.append(sanitize_entry(entry, use_gpt))

    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"✅ Saved sanitized data to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original JSON")
    parser.add_argument("--output", required=True, help="Path to write cleaned JSON")
    parser.add_argument("--use_gpt", action="store_true", help="Use GPT-4o-mini for additional PII scrub")
    args = parser.parse_args()

    sanitize_dataset(args.input, args.output, args.use_gpt)