# from datasets import load_dataset

# def load_meta_corpus(file_path):
#     """
#     Load corpus from a JSONL file.
#     """
#     return load_dataset("json", data_files=file_path, split="train").to_list()

from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent  # Source/
DATA_PATH = BASE_DIR / "data" / "new_chunked_data.jsonl"

def load_meta_corpus():
    return load_dataset(
        "json",
        data_files=str(DATA_PATH),
        split="train"
    ).to_list()
