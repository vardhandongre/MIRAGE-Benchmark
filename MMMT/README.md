# MMMT (Multi-Modal Multi-Turn) Benchmark

<div align="center">
  <img src="../assets/multiturn_logo.png" alt="MMMT Logo" width="300"/>
</div>

MMMT is a benchmark framework for evaluating and testing multi-modal, multi-turn conversational VLM models. The framework supports both text and image inputs, enabling comprehensive testing of model's ability to handle complex, multi-turn conversations with visual context.

## Features

- Multi-modal conversation support (text + images)
- Multi-turn dialog handling
- Support for multiple LLM providers (OpenAI, Anthropic, TogetherAI)
- Task generation and evaluation framework
- Clarification vs. Response decision making
- Comprehensive validation and testing utilities

## Project Structure

```
MMMT/
├── baselines/     # Baseline models and implementations
├── data/         # Dataset and data processing utilities
├── evaluate/     # Evaluation scripts and metrics
├── src/          # Core source code
│   ├── llm_agents.py        # LLM client implementations
│   ├── generate_tasks.py    # Task generation utilities
│   ├── validator.py         # Validation functions
│   ├── prompt_builder.py    # Prompt construction utilities
│   └── data_utils.py        # Data processing utilities
└── utils/        # Utility functions and helpers
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd MMMT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export TOGETHER_API_KEY="your-together-key"  # If using TogetherAI
export HUGGINGFACE_TOKEN="your-hf-token"     # If using HuggingFace models
```

## Usage

### Basic Usage

```python
from src.llm_agents import get_llm_client

# Initialize an LLM client
llm = get_llm_client("openai", "gpt-4-vision-preview")

# Example conversation with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "path/to/image.jpg"
                }
            }
        ]
    }
]

response = llm.generate(messages)
```

### Task Generation

```python
from src.generate_tasks import generate_task

task = generate_task(
    entry_id="task_001",
    title="Plant Disease Identification",
    dialog_context=conversation_history,
    revealed_fact="The leaves have yellow spots",
    image_descriptions=["Image of tomato plant with yellow spots"],
    reasoning_mode="detailed"
)
```

## Supported Models

- OpenAI: GPT-4 Vision, GPT-4
- Anthropic: Claude 3 Opus, Claude 3 Sonnet
- TogetherAI: Various models including Llama 3.2/3.3, Qwen, DeepSeek

## Evaluation

The framework includes comprehensive evaluation tools to assess:
- Decision Accuracy (Clarify vs. Respond)
- Goal relevance
- Utterance Quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0).

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

See the [LICENSE](../LICENSE) file for details.

## Citation

[Add citation information if applicable]
