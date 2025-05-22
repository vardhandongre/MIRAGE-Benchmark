# MIRAGE-MultiTurn Task

The AgriAsk-MultiTurn Task is designed to evaluate language models' ability to handle multi-turn agricultural conversations, specifically focusing on the "clarify or respond" decision-making process.

### Task Description

In this task, models are presented with a dialog context from an agricultural conversation and must decide whether to:
1. Ask for clarification when the context is insufficient
2. Provide a direct response when enough information is available

The task is generated from real agricultural Q&A data, where each example includes:
- A dialog context (conversation history)
- A revealed fact (the next user message)
- Image descriptions (when available)
- A decision (clarify or respond)
- A clarification question or response based on the decision

### Data Generation

The task generation process:
1. Takes input from `multi_turn_qa_data.json`
2. Processes conversations to create appropriate context windows
3. Uses GPT-4 to generate the clarify-or-respond decisions and content
4. Outputs the generated tasks in batches to `data/generated_clarify_or_respond.json`

### Evaluation

Tasks are evaluated using an LLM-based judge that assesses:
- Decision quality (appropriateness of clarify vs. respond choice)
- Goal relevance (alignment with the conversation's objectives)

### Project Structure

```
agriask_multiturn_clarify_or_respond/
├── data/                    # Input and output data files
├── src/                     # Source code
│   ├── prompt_builder.py    # Builds prompts for task generation
│   ├── clarification_generator.py  # Handles clarification generation
│   └── sanitize_outputs.py  # Cleans and validates generated outputs
└── generate_task.py         # Main task generation script
```

### Usage

To generate tasks:
1. Set the `OPENAI_API_KEY` environment variable
2. Run `python generate_task.py`
3. Generated tasks will be saved in `data/generated_clarify_or_respond.json`

The script processes the input data in batches and saves intermediate results, with any failures logged to `task_failures.log`.
