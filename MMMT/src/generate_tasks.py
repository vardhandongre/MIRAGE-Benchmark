import os
import json
import uuid
import argparse
from tqdm import tqdm
from datasets import Dataset
from src.prompt_builder import build_prompt
from src.clarification_generator import extract_json

# Add our LLM agents module
from src.llm_agents import get_llm_client

# --- Config ---
DEFAULT_BATCH_SAVE_SIZE = 100
BATCH_FOLDER = "data/batches"
os.makedirs(BATCH_FOLDER, exist_ok=True)

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="data/source/filtered_multi_turn_qa_data.json")
parser.add_argument("--output", type=str, default="data/generated/generated_clarify_or_respond_filtered.json")
parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (claude-3-7-sonnet, llama3-1-70b-instruct, etc.)")
parser.add_argument("--provider", type=str, default="openai", help="Provider: aws-bedrock (chat interface) or bedrock (standard)")
parser.add_argument("--start_index", type=int, default=0, help="Start index in source data")
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SAVE_SIZE)
parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of tasks to generate")
parser.add_argument("--reasoning_mode", action="store_true", help="Enable chain-of-thought prompting")
args = parser.parse_args()

# Create LLM client based on provider
try:
    if args.provider == "bedrock":
        # Chat interface version with history
        llm = get_llm_client("bedrock", args.model, temperature=0.0)
    else:
        # Standard message interface
        llm = get_llm_client("openai", args.model, temperature=0.0)
except Exception as e:
    print(f"Error initializing LLM client: {e}")
    print("Make sure AWS credentials are set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")
    exit(1)

# --- Functions ---
def detect_speaker(responder, idx):
    responder = responder.lower()
    if "question asker" in responder:
        return "user"
    return "expert"

def build_dialog_context(item):
    conversation = [{"speaker": "user", "text": item["question"]}]
    for i, r in enumerate(item.get("responses", [])):
        speaker = detect_speaker(r["responder"], i)
        conversation.append({"speaker": speaker, "text": r["response"]})
    return conversation

def generate_task(entry_id, title, dialog_context, revealed_fact, image_descriptions, reasoning_mode):
    """Generate task using either chat interface or standard messages interface"""
    prompt = build_prompt(dialog_context, [revealed_fact], image_descriptions, reasoning_mode=reasoning_mode)
    
    try:
        if args.provider == "aws-bedrock" and hasattr(llm, 'chat'):
            # Use chat interface
            raw_output = llm.chat(prompt, temperature=0.0)
        else:
            # Use standard messages interface
            messages = [{"role": "user", "content": prompt}]
            raw_output = llm.generate(messages, temperature=0.0)
    except Exception as e:
        raise Exception(f"Error generating task: {str(e)}")
    
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
    with open(args.input) as f:
        data = json.load(f)

    output = []
    total_generated = 0
    failed = []
    batch_buffer = []

    print(f"Using {args.provider} provider with model {args.model}")
    
    for item in tqdm(data[args.start_index:], desc="Generating tasks", initial=args.start_index, total=len(data) - args.start_index):
        if total_generated >= args.max_samples:
            break
            
        try:
            conversation = build_dialog_context(item)
            user_turns = [t for t in conversation if t["speaker"] == "user"]

            if len(user_turns) < 2:
                continue

            for i in range(1, len(user_turns)):
                if total_generated >= args.max_samples:
                    break
                    
                current_user_text = user_turns[i - 1]["text"]
                revealed_fact = user_turns[i]["text"]
                idx = next(j for j, t in enumerate(conversation) if t["text"] == current_user_text and t["speaker"] == "user")
                dialog_context = conversation[:idx + 1]

                if dialog_context[-1]["speaker"] != "user":
                    continue

                image_descriptions = item.get("attachments", [])
                task = generate_task(
                    entry_id=item["id"],
                    title=item["title"],
                    dialog_context=dialog_context,
                    revealed_fact=revealed_fact,
                    image_descriptions=image_descriptions,
                    reasoning_mode=args.reasoning_mode
                )
                output.append(task)
                batch_buffer.append(task)
                total_generated += 1

                # Save batch when buffer reaches batch_size
                if len(batch_buffer) >= args.batch_size:
                    batch_id = uuid.uuid4().hex[:8]
                    batch_path = os.path.join(BATCH_FOLDER, f"batch_{batch_id}.json")
                    with open(batch_path, "w") as f:
                        json.dump(batch_buffer, f, indent=2)
                    print(f" Saved {args.batch_size} samples to {batch_path}")
                    batch_buffer = []

        except Exception as e:
            failed.append({"id": item.get("id"), "error": str(e)})
            print(f"⚠️ Error processing item {item.get('id')}: {str(e)}")
            continue

    # Save any remaining items in the buffer
    if batch_buffer:
        batch_id = uuid.uuid4().hex[:8]
        batch_path = os.path.join(BATCH_FOLDER, f"batch_{batch_id}.json")
        with open(batch_path, "w") as f:
            json.dump(batch_buffer, f, indent=2)
        print(f"💾 Saved final {len(batch_buffer)} samples to {batch_path}")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f" Saved {len(output)} total examples to {args.output}")

    if failed:
        with open("task_failures.log", "w") as f:
            for entry in failed:
                f.write(json.dumps(entry) + "\n")
        print(f"⚠️ Logged {len(failed)} failures.")
    
    # If using chat interface, show conversation history
    if args.provider == "aws-bedrock" and hasattr(llm, 'get_history'):
        history = llm.get_history()
        print(f"\n Conversation history has {len(history)} entries")

if __name__ == "__main__":
    run_generation()