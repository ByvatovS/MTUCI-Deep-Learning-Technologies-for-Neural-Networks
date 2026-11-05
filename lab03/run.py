"""Лабораторная 3: маленькая decoder-only языковая модель на PyTorch.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
DEFAULT_BPE = HERE.parent / "lab02" / "artifacts" / "bpe.json"
LOCAL_CORPUS = HERE / "corpus.txt"
LAB02_CORPUS = HERE.parent / "lab02" / "corpus.txt"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")
PROMPT = "князь андрей"


# -------------------------------------------------------------------------- DATA
def save_json(name: str, value: dict) -> None:
    (ARTIFACTS / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_corpus() -> str:
    """Берёт уже скачанный корпус лабы 2 либо скачивает свою копию."""
    if LAB02_CORPUS.exists():
        return LAB02_CORPUS.read_text(encoding="utf-8")
    if not LOCAL_CORPUS.exists():
        print("Скачиваю корпус «Война и мир»...")
        urllib.request.urlretrieve(CORPUS_URL, LOCAL_CORPUS)
    return LOCAL_CORPUS.read_text(encoding="utf-8")


def normalise_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)


def apply_bpe(word: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    """Применяет слияния BPE в том же порядке, в котором они обучались."""
    symbols = tuple(word) + ("</w>",)
    while len(symbols) > 1:
        candidates = [
            (ranks[pair], pos)
            for pos, pair in enumerate(zip(symbols, symbols[1:]))
            if pair in ranks
        ]
        if not candidates:
            break
        _, position = min(candidates)
        symbols = merge_pair(symbols, (symbols[position], symbols[position + 1]))
    return list(symbols)


class BPETokenizer:
    def __init__(self, merges: list[list[str]], words: list[str]):
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.cache: dict[str, list[str]] = {}
        for word in set(words):
            self.cache[word] = apply_bpe(word, self.ranks)
        token_set = {piece for pieces in self.cache.values() for piece in pieces}
        self.id_to_token = sorted(token_set)
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def pieces(self, word: str) -> list[str]:
        if word not in self.cache:
            self.cache[word] = apply_bpe(word, self.ranks)
        return self.cache[word]

    def encode_words(self, words: list[str]) -> list[int]:
        return [self.token_to_id[piece] for word in words for piece in self.pieces(word)]

    def encode_text(self, text: str) -> list[int]:
        return self.encode_words(normalise_words(text))

    def decode(self, ids: list[int]) -> str:
        raw = "".join(self.id_to_token[index] for index in ids)
        return re.sub(r"\s+", " ", raw.replace("</w>", " ")).strip()

# -------------------------------------------------------------------------- MODEL
class CausalSelfAttention(nn.Module):
    """Многоголовое causal attention, записанное по формуле без готового слоя."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model должен делиться на n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = (weights @ value).transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(attended)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads)
        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attn(x))
        return x + self.feed_forward(self.norm_ff(x))


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, context: int):
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, length = ids.shape
        if length > self.context:
            raise ValueError(f"контекст {length} превышает {self.context}")
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


# -------------------------------------------------------------------------- TRAIN
    def make_batch(tokens: torch.Tensor, batch: int, context: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(tokens) - context - 1, (batch,), generator=generator)
        offsets = torch.arange(context)
        indices = starts[:, None] + offsets[None, :]
        return tokens[indices], tokens[indices + 1]


    @torch.no_grad()
    def mean_loss(model: TinyDecoder, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        model.eval()
        values = [F.cross_entropy(model(x).reshape(-1, model.lm_head.out_features), y.reshape(-1)).item() for x, y in batches]
        model.train()
        return float(sum(values) / len(values))


# -------------------------------------------------------------------------- GENERATION
def sample_next(logits: torch.Tensor, temperature: float, top_p: float | None) -> int:
    if temperature == 0:
        return int(torch.argmax(logits).item())
    probabilities = F.softmax(logits / temperature, dim=-1)
    if top_p is not None:
        sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
        keep[0] = True
        filtered = sorted_probs * keep
        filtered = filtered / filtered.sum()
        return int(sorted_ids[torch.multinomial(filtered, 1)].item())
    return int(torch.multinomial(probabilities, 1).item())


@torch.no_grad()
def generate(model: TinyDecoder, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int,
            temperature: float = 1.0, top_p: float | None = None) -> tuple[str, list[int]]:
    model.eval()
    current = tokenizer.encode_text(prompt)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        window = torch.tensor([current[-model.context:]], dtype=torch.long)
        next_id = sample_next(model(window)[0, -1], temperature, top_p)
        current.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(current), generated


def first_repeat(ids: list[int], width: int = 4) -> int | None:
    seen: set[tuple[int, ...]] = set()
    for end in range(width, len(ids) + 1):
        group = tuple(ids[end - width:end])
        if group in seen:
            return end - width
        seen.add(group)
    return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    parser.add_argument("--bpe", type=Path, default=DEFAULT_BPE, help="путь к bpe.json из лабы 2")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    raw_words = normalise_words(get_corpus())
    split = int(len(raw_words) * 0.9)
    train_words, val_words = raw_words[:split], raw_words[split:]
    bpe_data = json.loads(args.bpe.read_text(encoding="utf-8"))
    tokenizer = BPETokenizer(bpe_data["merges"], raw_words)
    train_tokens = torch.tensor(tokenizer.encode_words(train_words), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode_words(val_words), dtype=torch.long)

    if args.fast:
        d_model, n_layers, n_heads, d_ff, context, steps, batch, report_every = 64, 2, 4, 256, 128, 200, 16, 25
    else:
        d_model, n_layers, n_heads, d_ff, context, steps, batch, report_every = 128, 4, 4, 512, 128, 2000, 32, 100
    # У fast меньше обновлений, поэтому ему нужен немного более крупный шаг,
    # чтобы к 200-му шагу кривая успела показать заметное обучение.
    lr = 2e-3 if args.fast else 1e-3
    model = TinyDecoder(tokenizer.vocab_size, d_model, n_layers, n_heads, d_ff, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(parameter.numel() for parameter in model.parameters())


    val_generator = torch.Generator().manual_seed(seed + 1)
    val_batches = [make_batch(val_tokens, batch, context, val_generator) for _ in range(20)]
    probe_generator = torch.Generator().manual_seed(seed + 2)
    train_probe = [make_batch(train_tokens, batch, context, probe_generator) for _ in range(4)]
    curve: list[dict[str, float | int]] = [{"step": 0, "train_loss": mean_loss(model, train_probe), "val_loss": mean_loss(model, val_batches)}]
    sample_steps = [0, 50, 100, steps] if args.fast else [0, 100, 500, steps]
    torch.manual_seed(seed + 10)
    samples = [{"step": 0, "text": generate(model, tokenizer, PROMPT, 60, temperature=0.8)[0]}]

    print(f"Токенов: train={len(train_tokens)}, val={len(val_tokens)}; словарь={tokenizer.vocab_size}")
    print(f"Модель: {n_params:,} параметров; обучение {steps} шагов на CPU...")
    started = time.perf_counter()
    running_loss = 0.0
    for step in range(1, steps + 1):
        x, y = make_batch(train_tokens, batch, context, generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += float(loss.item())
        if step in sample_steps and step != 0:
            torch.manual_seed(seed + 10 + step)
            samples.append({"step": step, "text": generate(model, tokenizer, PROMPT, 60, temperature=0.8)[0]})
        if step % report_every == 0 or step == steps:
            curve.append({"step": step, "train_loss": running_loss / report_every, "val_loss": mean_loss(model, val_batches)})
            print(f"шаг {step:4d}: train={curve[-1]['train_loss']:.4f}, val={curve[-1]['val_loss']:.4f}")
            running_loss = 0.0
    minutes = (time.perf_counter() - started) / 60

    with (ARTIFACTS / "train_curve.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(curve)
    save_json("tiny_config.json", {
        "vocab_size": tokenizer.vocab_size, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "d_ff": d_ff, "context": context, "n_params": n_params, "steps": steps, "batch": batch, "lr": lr,
        "seed": seed, "train_tokens": len(train_tokens), "val_tokens": len(val_tokens),
        "bpe_merges": len(bpe_data["merges"]), "train_minutes": round(minutes, 3), "device": "cpu",
    })
    save_json("samples.json", {"prompt": PROMPT, "temperature": 0.8, "samples": samples})

    max_new_tokens = 40 if args.fast else 80
    modes = [("greedy", 0.0, None), ("t0.7", 0.7, None), ("t1.5", 1.5, None), ("top_p0.9", 1.0, 0.9)]
    runs = []
    for name, temperature, top_p in modes:
        torch.manual_seed(seed + 100 + len(runs))
        text, ids = generate(model, tokenizer, PROMPT, max_new_tokens, temperature, top_p)
        run = {"name": name, "temperature": temperature, "top_p": top_p, "text": text, "ids": ids, "n_tokens": len(ids)}
        if name == "greedy":
            run["loop_start"] = first_repeat(ids)
        runs.append(run)
    save_json("generation.json", {"model": "tiny", "prompt": PROMPT, "max_new_tokens": max_new_tokens, "runs": runs})
    print(f"Готово за {minutes:.2f} мин. Артефакты: {ARTIFACTS}")


if __name__ == "__main__":
    main()
