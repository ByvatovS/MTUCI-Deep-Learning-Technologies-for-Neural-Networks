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


# ------------------------------------------------------------------------- Search
def select_paragraphs(corpus: str, fast: bool) -> list[dict]:
    raw_paragraphs = re.split(r"\r?\n\s*\r?\n", corpus)
    selected = []
    for raw in raw_paragraphs:
        text = re.sub(r"\s+", " ", raw).strip()
        if 40 <= len(normalized_words(text)) <= 120:
            selected.append({"id": len(selected), "text": text})
    return selected[:(60 if fast else 500)]


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


def cosine_top_k(matrix, queries, np, k: int = 5) -> list[list[int]]:
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-12)
    return [np.argsort(-(row), kind="stable")[:k].astype(int).tolist() for row in queries @ matrix.T]


def make_search_artifacts(corpus: str, fast: bool, qwen_tokenizer, qwen_model, torch, np, AutoModel, AutoTokenizer):
    paragraphs = select_paragraphs(corpus, fast)
    if not paragraphs:
        raise RuntimeError("Не удалось выбрать абзацы из корпуса")
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    texts, query_texts = [p["text"] for p in paragraphs], [q["query"] for q in queries]
    print(f"Строю векторы Qwen для {len(paragraphs)} абзацев...")
    static_top = cosine_top_k(qwen_vectors(texts, qwen_tokenizer, qwen_model, "static", torch),
                              qwen_vectors(query_texts, qwen_tokenizer, qwen_model, "static", torch), np)
    contextual_top = cosine_top_k(qwen_vectors(texts, qwen_tokenizer, qwen_model, "contextual", torch),
                                  qwen_vectors(query_texts, qwen_tokenizer, qwen_model, "contextual", torch), np)
    print("Загружаю multilingual-e5-small и строю поисковые векторы...")
    e5_tokenizer, e5_model = AutoTokenizer.from_pretrained(E5_MODEL), AutoModel.from_pretrained(E5_MODEL)
    e5_model.eval()
    e5_top = cosine_top_k(e5_vectors(texts, "passage: ", e5_tokenizer, e5_model, torch),
                          e5_vectors(query_texts, "query: ", e5_tokenizer, e5_model, torch), np)
    result_queries = []
    for index, source in enumerate(queries):
        matches = [p["id"] for p in paragraphs if source["answer_phrase"] in p["text"]]
        if not matches:
            raise RuntimeError(f"В выбранных абзацах нет ответа: {source['answer_phrase']}")
        result_queries.append({**source, "answer_id": matches[0],
                               "top": {"static": static_top[index], "contextual": contextual_top[index], "e5": e5_top[index]}})
    recalls = {method: round(sum(q["answer_id"] in q["top"][method] for q in result_queries) / len(result_queries), 3)
               for method in ("static", "contextual", "e5")}
    return paragraphs, {"k": 5, "n_paragraphs": len(paragraphs), "queries": result_queries, "recall_at_k": recalls}



# ---------------------------------------------------------------------- Attention
def token_positions(offsets, start: int, end: int) -> list[int]:
    return [i for i, (left, right) in enumerate(offsets) if left < end and right > start]


def attention_artifact(tokenizer, model, torch) -> dict:
    encoded = tokenizer(ATTENTION_SENTENCE, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    offsets, input_ids = encoded.pop("offset_mapping")[0].tolist(), encoded["input_ids"]
    pronoun, antecedent = re.search(r"\bего\b", ATTENTION_SENTENCE), re.search(r"пациента", ATTENTION_SENTENCE)
    assert pronoun and antecedent
    pronoun_pos = token_positions(offsets, pronoun.start(), pronoun.end())
    antecedent_pos = token_positions(offsets, antecedent.start(), antecedent.end())
    with torch.inference_mode():
        attentions = model(**encoded, output_attentions=True, use_cache=False).attentions
    best_share, best_layer, best_head, best_weights = -1.0, 0, 0, None
    for layer, layer_attention in enumerate(attentions):
        rows = layer_attention[0, :, pronoun_pos[-1], :].float().cpu()
        value, head = rows[:, antecedent_pos].sum(dim=1).max(dim=0)
        if float(value) > best_share:
            best_share, best_layer, best_head = float(value), layer, int(head)
            best_weights = rows[best_head].tolist()
    assert best_weights is not None
    return {"sentence": ATTENTION_SENTENCE, "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "pronoun_pos": pronoun_pos, "antecedent_pos": antecedent_pos, "best_layer": best_layer,
            "best_head": best_head, "weights": best_weights, "share_on_antecedent": round(best_share, 6)}


def draw_attention_chart(attention: dict) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(max(10, len(attention["tokens"]) * .55), 4.8))
    colors = ["#d95f02" if i in attention["antecedent_pos"] else "#4c78a8" for i in range(len(attention["tokens"]))]
    plt.bar(range(len(attention["tokens"])), attention["weights"], color=colors)
    plt.xticks(range(len(attention["tokens"])), attention["tokens"], rotation=45, ha="right")
    plt.ylabel("Вес внимания")
    plt.title(f"Qwen3: слой {attention['best_layer']}, голова {attention['best_head']}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "attention_map.png", dpi=160)
    plt.close()


def dilution_rows(corpus: str, tokenizer, model, torch, fast: bool) -> list[dict]:
    lengths = (64, 256, 768) if fast else (128, 512, 1024)
    instruction_ids = tokenizer(INSTRUCTION, add_special_tokens=False)["input_ids"]
    newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    filler_ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]
    rows = []
    for context_len in lengths:
        ids = (instruction_ids + newline_ids + filler_ids)[:context_len]
        if len(ids) != context_len:
            raise RuntimeError("Корпус слишком короток для опыта с attention")
        inputs = {"input_ids": torch.tensor([ids], dtype=torch.long),
                  "attention_mask": torch.ones((1, context_len), dtype=torch.long)}
        with torch.inference_mode():
            layers = model(**inputs, output_attentions=True, use_cache=False).attentions
        instruction_end = len(instruction_ids)
        shares = [layer[0, :, -1, 1:instruction_end].float().sum(dim=-1).mean().item() for layer in layers]
        sink = [layer[0, :, -1, 0].float().mean().item() for layer in layers]
        instruction_len = instruction_end - 1  # первый токен намеренно исключён как attention sink
        rows.append({"context_len": context_len, "instruction_len": instruction_len,
                     "share_on_instruction": round(sum(shares) / len(shares), 6),
                     "sink_share": round(sum(sink) / len(sink), 6),
                     "uniform_share": round(instruction_len / context_len, 6)})
    return rows



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

    paragraphs, search = make_search_artifacts(corpus, args.fast, qwen_tokenizer, qwen_model, torch, np, AutoModel, AutoTokenizer)
    save_json("paragraphs.json", paragraphs)
    save_json("search.json", search)

    attention = attention_artifact(qwen_tokenizer, qwen_model, torch)
    save_json("attention_map.json", attention)
    draw_attention_chart(attention)
    dilution = dilution_rows(corpus, qwen_tokenizer, qwen_model, torch, args.fast)
    with (ARTIFACTS / "dilution.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dilution[0]))
        writer.writeheader()
        writer.writerows(dilution)


if __name__ == "__main__":
    main()
