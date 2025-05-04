import os

MODEL_NAME = "gpt-4o"
BATCH_SIZE = 500
HF_REPO = "vdongre2/AgriAsk"
CONFIG_NAME = "multiturn_clarify_or_respond"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")