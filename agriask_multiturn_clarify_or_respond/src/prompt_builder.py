def build_prompt(dialog_context, revealed_facts, image_descriptions):
    dialog_text = "\n".join([f"{t['speaker'].capitalize()}: {t['text']}" for t in dialog_context])
    facts_list = "\n- ".join(revealed_facts)
    image_note = "\n".join([f"- {desc}" for desc in image_descriptions]) if image_descriptions else ""

    return f"""
You are an expert in agriculture and horticulture answering questions in a multi-turn conversation.

Your goal is to decide:
- <Clarify>: if you need more context to answer usefully, ask a goal-relevant clarification question.
- <Respond>: if the user has already provided enough information, provide a helpful response.

Later in the conversation, the user shared these facts:
- {facts_list}

Image context (if applicable):
{image_note}

Here is the dialog so far:
\"\"\"
{dialog_text}
\"\"\"

Return this JSON:
{{
  "decision": "<Clarify>" or "<Respond>",
  "goal": "...",
  "goal_state": {{
    "known": {{...}},
    "missing": [...]
  }},
  "clarification_question": "...",  # required if decision == <Clarify>
  "response": "..."                 # required if decision == <Respond>
}}
"""
