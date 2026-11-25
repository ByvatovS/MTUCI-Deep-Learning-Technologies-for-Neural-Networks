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

from common import (
    ADAPTER, ARTIFACTS, DATASET_NAME, MODEL_NAME, SEED, chat_ids,
    device_and_dtype, few_shot_examples, load_reviews, messages_for, parse_star, 
)

#---------------------------------------------- GENERATION AND PARSING
@torch.inference_mode()
def predict(model, tokenizer, rows, examples, device: torch.device) -> tuple[list[int | None], list[str], float, float]:
    """Жадная генерация не более трёх токенов и разбор только цифры 1–5."""
    predictions: list[int | None] = []
    raw_answers: list[str] = []
    prompt_tokens = 0
    started = time.perf_counter()
    model.eval()
    for row in rows:
        ids = chat_ids(tokenizer, messages_for(row["text"], examples), add_generation_prompt=True)
        prompt_tokens += len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids=input_ids, max_new_tokens=3, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )[0, len(ids):]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        raw_answers.append(raw)
        predictions.append(parse_star(raw))
    elapsed = time.perf_counter() - started
    return predictions, raw_answers, elapsed / len(rows), prompt_tokens / len(rows)




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
