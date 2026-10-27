"""Лабораторная 2: токенизация, представления текста и attention.

Запуск из корня проекта или из любой другой папки:
    python lab02/run.py --fast
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS_PATH = HERE / "corpus.txt"
COURSE_ROOT = HERE.parent
QUERIES_PATH = COURSE_ROOT / "tests" / "data" / "lab02" / "queries.json"
CORPUS_URL = ("https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
              "master/books_before/WarAndPeace.txt")
QWEN_MODEL = "Qwen/Qwen3-0.6B-Base"
E5_MODEL = "intfloat/multilingual-e5-small"
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")
ENGLISH_PARAGRAPH = (
    "The prince had been talking for almost an hour, and the guests were growing tired. "
    "Anna looked at the clock, then at the window, then at the clock again. Nobody dared "
    "to leave before the host, and the host did not seem to notice the time at all. A servant "
    "brought tea, and the conversation turned, as it always did in that house, to the war and "
    "to the emperor."
)
ATTENTION_SENTENCE = "Врач осмотрел пациента и сказал, что его анализы готовы."
INSTRUCTION = "Отвечай только по-русски и в конце каждого ответа ставь слово КОНЕЦ."


def save_json(name: str, payload: object) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_words(text: str) -> list[str]:
    """Слова ровно по правилу задания: нижний регистр и ё -> е."""
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def load_corpus() -> str:
    """Скачивает корпус один раз; сам corpus.txt не должен попадать в Git."""
    if not CORPUS_PATH.exists():
        print("Скачиваю корпус «Война и мир»...")
        request = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "lab02-course-script"})
        with urllib.request.urlopen(request, timeout=60) as response:
            CORPUS_PATH.write_bytes(response.read())
    return CORPUS_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- BPE
def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def train_bpe(words: Iterable[str], n_merges: int, min_frequency: int = 2) -> list[tuple[str, str]]:
    """Классический BPE над словарём типов; частоты учитываются в каждой паре."""
    frequencies = Counter(words)
    vocab: dict[tuple[str, ...], int] = {
        tuple(word) + ("</w>",): count for word, count in frequencies.items() if count >= min_frequency
    }
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for symbols, count in vocab.items():
            # Одна и та же пара может встретиться в слове несколько раз, поэтому
            # нельзя передавать dict в Counter.update: он бы потерял повторы.
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break
        best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
        merges.append(best_pair)
        next_vocab: dict[tuple[str, ...], int] = {}
        for symbols, count in vocab.items():
            new_symbols = merge_pair(symbols, best_pair)
            next_vocab[new_symbols] = next_vocab.get(new_symbols, 0) + count
        vocab = next_vocab
    return merges


def apply_bpe(word: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    """Применяет слияния в порядке обучения (как проверяют тесты)."""
    symbols = list(word) + ["</w>"]
    while len(symbols) > 1:
        candidates = [(ranks[pair], i) for i, pair in enumerate(zip(symbols, symbols[1:])) if pair in ranks]
        if not candidates:
            break
        _, index = min(candidates)
        symbols = symbols[:index] + [symbols[index] + symbols[index + 1]] + symbols[index + 2:]
    return symbols


def make_bpe_artifact(words: list[str], fast: bool) -> dict:
    n_merges = 200 if fast else 1500
    print(f"Обучаю BPE ({n_merges} слияний)...")
    merges = train_bpe(words, n_merges=n_merges)
    if len(merges) < n_merges:
        raise RuntimeError("В корпусе недостаточно различных пар для BPE")
    ranks = {pair: index for index, pair in enumerate(merges)}
    frequencies = Counter(words)
    token_cache = {word: apply_bpe(word, ranks) for word in frequencies}
    tokens_bpe = sum(len(token_cache[word]) * count for word, count in frequencies.items())
    chars = sum(len(word) * count for word, count in frequencies.items())
    examples = {word: token_cache.get(word, apply_bpe(word, ranks))
                for word in ("наполеон", "андрей", "сказала", "неудовольствие", "лошадь")}
    return {
        "vocab_size": len(set(symbol for tokens in token_cache.values() for symbol in tokens)),
        "num_merges": len(merges), "merges": [list(pair) for pair in merges],
        "stats": {"chars": chars, "words": len(words), "tokens_bpe": tokens_bpe,
                  "tokens_per_word": round(tokens_bpe / len(words), 3),
                  "compression_vs_chars": round(chars / tokens_bpe, 3)},
        "examples": examples,
    }


# --------------------------------------------------------------------- Transformers
def import_ml():
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Нужны зависимости из requirements.txt: pip install -r requirements.txt") from exc
    return np, torch, AutoModel, AutoTokenizer


def load_qwen(AutoModel, AutoTokenizer):
    print("Загружаю Qwen3-0.6B-Base...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # eager обязателен для output_attentions; модель не нужно загружать дважды.
    model = AutoModel.from_pretrained(QWEN_MODEL, attn_implementation="eager")
    model.eval()
    return tokenizer, model


def masked_mean(hidden, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def qwen_vectors(texts: list[str], tokenizer, model, kind: str, torch, batch_size: int = 4):
    """Static = таблица input embeddings; contextual = последний hidden state."""
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                              truncation=True, max_length=512, add_special_tokens=False)
            if kind == "static":
                hidden = model.get_input_embeddings()(batch["input_ids"])
            else:
                hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states[-1]
            vectors.append(masked_mean(hidden, batch["attention_mask"], torch).float().cpu())
    return torch.cat(vectors).numpy()


def e5_vectors(texts: list[str], prefix: str, tokenizer, model, torch, batch_size: int = 16):
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer([prefix + text for text in texts[start:start + batch_size]],
                              return_tensors="pt", padding=True, truncation=True, max_length=512)
            vectors.append(masked_mean(model(**batch).last_hidden_state, batch["attention_mask"], torch).float().cpu())
    return torch.cat(vectors).numpy()


def make_token_artifact(corpus_words: list[str], tokenizer) -> dict:
    ru_words = corpus_words[499:599]  # слова с 500 по 600, ровно 100
    ru_text = " ".join(ru_words)
    en_words = ENGLISH_PARAGRAPH.split()
    ru_ids = tokenizer(ru_text, add_special_tokens=False)["input_ids"]
    en_ids = tokenizer(ENGLISH_PARAGRAPH, add_special_tokens=False)["input_ids"]
    ru_per_word, en_per_word = len(ru_ids) / len(ru_words), len(en_ids) / len(en_words)
    return {"tokenizer": QWEN_MODEL, "vocab_size": int(tokenizer.vocab_size), "context_tokens": 4096,
            "ru": {"words": len(ru_words), "tokens": len(ru_ids), "tokens_per_word": round(ru_per_word, 3),
                   "words_in_context": int(4096 / ru_per_word)},
            "en": {"words": len(en_words), "tokens": len(en_ids), "tokens_per_word": round(en_per_word, 3),
                   "words_in_context": int(4096 / en_per_word)},
            "example_ru": tokenizer.convert_ids_to_tokens(ru_ids[:20])}




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    corpus = load_corpus()
    words = normalized_words(corpus)
    save_json("bpe.json", make_bpe_artifact(words, args.fast))

    np, torch, AutoModel, AutoTokenizer = import_ml()
    qwen_tokenizer, qwen_model = load_qwen(AutoModel, AutoTokenizer)
    tokens = make_token_artifact(words, qwen_tokenizer)
    save_json("tokens_per_word.json", tokens)


if __name__ == "__main__":
    main()
