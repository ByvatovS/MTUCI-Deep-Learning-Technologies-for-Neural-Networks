"""Оценка zero-shot, few-shot и LoRA-адаптера для лабораторной 4.

Запуск: python lab04/run.py
Быстрый режим для CI: python lab04/run.py --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import torch



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="первые 24 тестовых отзыва")
    parser.add_argument("--data", help="папка dataset.save_to_disk либо переменная LAB_DATA")
    args = parser.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    device, dtype = device_and_dtype()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    dataset = load_reviews(args.data)
    limit = 24 if args.fast else 1000
    rows = dataset["test"].select(range(limit))
    labels = [int(row["label"]) + 1 for row in rows]
    few_examples = few_shot_examples(dataset["validation"])



if __name__ == "__main__":
    main()
