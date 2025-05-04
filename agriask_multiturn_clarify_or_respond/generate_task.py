import os
import json
import uuid
from tqdm import tqdm
import openai
from datasets import Dataset
from src.prompt_builder import build_prompt
from src.clarification_generator import extract_json

INPUT_FILE = "data/multi_turn_qa_data.json"
OUTPUT_FILE = "data/generated_clarify_or_respond.json"
BATCH_SIZE = 5
MODEL_NAME = "gpt-4o"
openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_speaker(responder, idx):
    responder = responder.lower()
    if "question asker" in responder:
        return "user"
    if idx == 0:
        return "expert"
    return "expert"

def build_dialog_context(item):
    conversation = [{"speaker": "user", "text": item["question"]}]
    for i, r in enumerate(item.get("responses", [])):
        speaker = detect_speaker(r["responder"], i)
        conversation.append({"speaker": speaker, "text": r["response"]})
    return conversation

def generate_task(entry_id, title, dialog_context, revealed_fact, image_descriptions):
    prompt = build_prompt(dialog_context, [revealed_fact], image_descriptions)
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0
    )
    raw_output = response['choices'][0]['message']['content'].strip()
    parsed = extract_json(raw_output)

    return {
        "task_type": "clarify_or_respond",
        "decision": parsed["decision"],
        "goal": parsed["goal"],
        "goal_state": parsed["goal_state"],
        "clarification_question": parsed.get("clarification_question"),
        "response": parsed.get("response"),
        "dialog_context": dialog_context,
        "gold_revealed_fact": revealed_fact,
        "image": image_descriptions,
        "source_id": entry_id,
        "title": title,
        "evaluation": {
            "type": "LLM-Judge",
            "criteria": ["decision quality", "goal relevance"]
        }
    }

def run_generation():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    output = []
    failed = []

    for item in tqdm(data, desc="Generating tasks"):
        try:
            conversation = build_dialog_context(item)
            user_turns = [t for t in conversation if t["speaker"] == "user"]

            if len(user_turns) < 2:
                continue  # Need at least two user messages for this setup

            # We take the context up to the second-last user turn
            for i in range(1, len(user_turns)):
                current_user_text = user_turns[i - 1]["text"]
                revealed_fact = user_turns[i]["text"]

                # Truncate context up to the current user turn
                idx = next(j for j, t in enumerate(conversation) if t["text"] == current_user_text and t["speaker"] == "user")
                dialog_context = conversation[:idx + 1]

                if dialog_context[-1]["speaker"] != "user":
                    continue  # We want context to end in user turn

                image_descriptions = item.get("attachments", [])
                task = generate_task(
                    entry_id=item["id"],
                    title=item["title"],
                    dialog_context=dialog_context,
                    revealed_fact=revealed_fact,
                    image_descriptions=image_descriptions
                )
                output.append(task)

                if len(output) % BATCH_SIZE == 0:
                    batch_id = uuid.uuid4().hex[:8]
                    with open(f"data/batch_{batch_id}.json", "w") as f:
                        json.dump(output, f, indent=2)

        except Exception as e:
            failed.append({"id": item.get("id"), "error": str(e)})
            continue

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Saved {len(output)} examples to {OUTPUT_FILE}")

    if failed:
        with open("task_failures.log", "w") as f:
            for entry in failed:
                f.write(json.dumps(entry) + "\n")
        print(f"⚠️ Logged {len(failed)} failures.")

if __name__ == "__main__":
    run_generation()