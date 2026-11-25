"""Общие функции для обучения и оценки лабораторной 4."""

from __future__ import annotations

import os
import random
import re
from collections import Counter
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ADAPTER = ARTIFACTS / "adapter"
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "mteb/GeoreviewClassification"
SEED = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SYSTEM_PROMPT = (
    "Ты классифицируешь отзыв об организации. Определи оценку, которую поставил "
    "автор, по шкале от 1 до 5: 1 — очень плохо, 2 — плохо, 3 — нейтрально, "
    "4 — хорошо, 5 — отлично. Отвечай только одной цифрой от 1 до 5."
)

# Делается до импорта transformers/datasets: Hugging Face читает этот флаг при
# импорте. На Colab без кэша флаг не ставится, и модель нормально скачается.
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
QWEN_SNAPSHOTS = HF_HOME / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots"
if QWEN_SNAPSHOTS.exists() and any(QWEN_SNAPSHOTS.iterdir()):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def model_load_kwargs() -> dict[str, bool]:
    """Не делает сетевых запросов, если Qwen уже есть в Hugging Face cache."""
    return {"local_files_only": True} if os.environ.get("HF_HUB_OFFLINE") else {}


#---------------------------------------------- DATA SELECTION
def load_reviews(data_path: str | None = None):
    """Загружает набор с Hub либо его ранее сохранённую локальную копию."""
    from datasets import Dataset, DatasetDict, config, load_dataset, load_from_disk

    source = data_path or os.environ.get("LAB_DATA")
    if source:
        return load_from_disk(source)
    try:
        return load_dataset(DATASET_NAME)
    except Exception:
        # После первого скачивания Hub может быть недоступен. В datasets 2.x
        # local_files_only всё равно запрашивает metadata, поэтому открываем
        # уже созданные Arrow-файлы непосредственно из кэша.
        files = list(Path(config.HF_DATASETS_CACHE).glob(
            "mteb___georeview_classification/**/georeview_classification-*.arrow"
        ))
        splits = {path.stem.rsplit("-", 1)[1]: Dataset.from_file(str(path)) for path in files}
        if {"train", "validation", "test"} <= splits.keys():
            return DatasetDict(splits)
        raise RuntimeError("Набор не найден в кэше. Подключитесь к сети или передайте --data.")


def balanced_train_indices(split, per_class: int, seed: int = SEED) -> list[int]:
    """Ровно per_class примеров каждого класса в воспроизводимом порядке."""
    generator = torch.Generator().manual_seed(seed)
    result: list[int] = []
    labels = split["label"]
    for label in range(5):
        candidates = [index for index, value in enumerate(labels) if value == label]
        order = torch.randperm(len(candidates), generator=generator).tolist()
        result.extend(candidates[index] for index in order[:per_class])
    random.Random(seed).shuffle(result)
    return result


def few_shot_examples(split) -> list[dict[str, object]]:
    """Первый пример каждого класса validation, общий для всех запросов."""
    examples: list[dict[str, object]] = []
    seen: set[int] = set()
    for row in split:
        label = int(row["label"])
        if label not in seen:
            examples.append({"text": row["text"], "label": label + 1})
            seen.add(label)
        if len(seen) == 5:
            return examples
    raise RuntimeError("В validation не нашлись все пять классов")


