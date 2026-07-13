import os

MODEL_DIR = "xxx"
BASE_DIR = "xxx"
DATA_DIR = "xxx"
DATA_PATH = os.path.join(DATA_DIR, "split_v1", "test.jsonl")
IMAGE_DIR = os.path.join(DATA_DIR, "img")
RESULT_DIR = {
    "base": os.path.join(BASE_DIR, "results", "base"),
    "ft": os.path.join(BASE_DIR, "results", "ft"),
    "test": os.path.join(BASE_DIR, "results", "test"),
    "circular_base": os.path.join(BASE_DIR, "results", "circular", "base"),
    "circular_ft": os.path.join(BASE_DIR, "results", "circular", "ft")
}
ROLE_PROMPT = (
    "You are currently a senior expert in sequence problems, focusing on specific research topics such as time, space, length, quantity, emotion, symmetry, logic, etc. "
    "You can sort their attributes, such as length, order, size, quantity, and strength. "
    "Given one or more images and a sorting problem, some questions can be answered directly, some questions require inference, and others can only choose relatively reasonable answers. "
    "Your task is to answer these questions from a human perspective and output the correct answers without any explanation. "
    "Please note that you only need to select one option from all options, and your response should only include the option letter (A, B, C, or D) and not any other text.\n"
)