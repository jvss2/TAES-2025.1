import os

DATASET_NAME = "claudios/dypybench_functions"
DATASET_SPLIT = "train"
MIN_EXECUTED_LINES = 2
SAMPLE_SIZE = None

LLM_BACKEND = "hf"  # "openai" or "hf"
LLM_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
HF_DEVICE = 0

MAX_TOKENS = 32
TEMPERATURE = 0
LOGPROBS = 1
PLACEHOLDER = "# [YOUR_LINE_HERE]"

OUTPUT_FILE = "results/dypybench_predictions"

INSTRUCTION_PROMPT = """Complete the Python function below by generating the next line of code.
Return ONLY the single next line that should come after the existing code.
Do not add any extra code, comments, or explanations beyond that single line.

"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")