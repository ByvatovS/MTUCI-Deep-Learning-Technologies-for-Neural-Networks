"""Тесты лабораторной 2: читают lab02/artifacts/ и пересчитывают из данных все, что можно пересчитать.

Скрипт студента тесты не запускают и не импортируют, артефакты к их запуску уже должны лежать на месте.
"""
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab02"
ART = LAB / "artifacts"
DATA = Path(__file__).resolve().parent / "data" / "lab02"
TOKENIZER = "Qwen/Qwen3-0.6B-Base"

# Пропуск через метку, а не через pytest.skip(allow_module_level=True): 
# тот не собирает ни одного теста, pytest выходит с кодом 5, и статус красный.
pytestmark = pytest.mark.skipif(
    "NotImplementedError" in (LAB / "run.py").read_text(encoding="utf-8"),
    reason="lab02/run.py это заглушка, работа не начата"
)


def load_json(path: Path):
    assert path.exists(), f"нет файла {path.relative_to(ROOT)}, запустите python lab02/run.py --fast"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def thresholds():
    return load_json(DATA / "thresholds.json")


# ---------------------------------------------------------------- bpe.json

@pytest.fixture(scope="module")
def bpe():
    return load_json(ART / "bpe.json")


def apply_merges(word: str, ranks: dict) -> list[str]:
    """Слова в буквы плюс </w>, на каждом шаге сливается пара с наименьшим рангом."""
    sym = list(word) + ["</w>"]
    while len(sym) > 1:
        ranked = [(ranks[pair], i) for i, pair in enumerate(zip(sym, sym[1:])) if pair in ranks]
        if not ranked:
            break
        _, i = min(ranked)
        sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
    return sym


def test_bpe_merges_count(bpe):
    assert len(bpe["merges"]) == bpe["num_merges"]
    assert bpe["num_merges"] >= 200


def test_bpe_examples_reproduce(bpe):
    ranks = {}
    for i, (a, b) in enumerate(bpe["merges"]):
        ranks.setdefault((a, b), i)
    for word in ("наполеон", "сказала", "неудовольствие"):
        assert word in bpe["examples"], f"в examples нет слова «{word}»"
    for word, tokens in bpe["examples"].items():
        assert apply_merges(word, ranks) == tokens, f"слияния по порядку дают для «{word}» другое разбиение"


def test_bpe_stats_plausible(bpe):
    assert 1.2 <= bpe["stats"]["tokens_per_word"] <= 4.0
    assert 1.3 <= bpe["stats"]["compression_vs_chars"] <= 4.0


# ---------------------------------------------------------------- tokens_per_word.json

@pytest.fixture(scope="module")
def tpw():
    return load_json(ART / "tokens_per_word.json")


def test_russian_costs_more(tpw):
    assert tpw["ru"]["tokens_per_word"] > tpw["en"]["tokens_per_word"]


def test_en_tokens_match_qwen(tpw):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    paragraph = (DATA / "en_paragraph.txt").read_text(encoding="utf-8").strip()
    expected = len(tok(paragraph, add_special_tokens=False)["input_ids"])
    assert abs(tpw["en"]["tokens"] - expected) <= 2, f"у токенизатора {TOKENIZER} абзац занимает {expected} токенов"


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_words_in_context_consistent(tpw, lang):
    expected = int(tpw["context_tokens"] / tpw[lang]["tokens_per_word"])
    assert tpw[lang]["words_in_context"] == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------- paragraphs.json, search.json

@pytest.fixture(scope="module")
def paragraphs():
    return load_json(ART / "paragraphs.json")


@pytest.fixture(scope="module")
def search():
    return load_json(ART / "search.json")


METHODS = ("static", "contextual", "e5")


def test_paragraph_selection(paragraphs):
    head = json.loads((DATA / "paragraphs_head.json").read_text(encoding="utf-8"))
    assert [p["text"] for p in paragraphs[: len(head)]] == head, "первые абзацы не совпадают с эталоном, проверьте правило отбора"


def test_queries_and_answers(paragraphs, search):
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    assert len(search["queries"]) == len(queries)
    for ref, got in zip(queries, search["queries"]):
        assert got["query"] == ref["query"]
        assert got["answer_phrase"] == ref["answer_phrase"]
        hits = [p["id"] for p in paragraphs if ref["answer_phrase"] in p["text"]]
        assert hits, f"фразы «{ref['answer_phrase']}» нет ни в одном абзаце"
        assert got["answer_id"] == hits[0], f"для «{ref['answer_phrase']}» первый абзац с фразой {hits[0]}"


@pytest.mark.parametrize("method", METHODS)
def test_top_shape(paragraphs, search, method):
    assert search["n_paragraphs"] == len(paragraphs)
    for q in search["queries"]:
        top = q["top"][method]
        assert len(top) == search["k"]
        assert all(0 <= i < search["n_paragraphs"] for i in top)


@pytest.mark.parametrize("method", METHODS)
def test_recall_recomputed(search, method):
    hits = sum(q["answer_id"] in q["top"][method] for q in search["queries"])
    assert search["recall_at_k"][method] == pytest.approx(hits / len(search["queries"]), abs=1e-3)


def test_e5_recall_threshold(search, thresholds):
    assert search["recall_at_k"]["e5"] >= thresholds["search_recall_e5"]


# ---------------------------------------------------------------- attention_map.json

@pytest.fixture(scope="module")
def attention():
    return load_json(ART / "attention_map.json")


def test_attention_is_distribution(attention):
    w = attention["weights"]
    assert len(w) == len(attention["tokens"])
    assert sum(w) == pytest.approx(1.0, abs=1e-3)
    assert min(w) >= 0


def test_attention_share_on_antecedent(attention, thresholds):
    assert attention["pronoun_pos"] and attention["antecedent_pos"]
    share = sum(attention["weights"][i] for i in attention["antecedent_pos"])
    assert attention["share_on_antecedent"] == pytest.approx(share, abs=1e-3)
    assert attention["share_on_antecedent"] >= thresholds["attention_share_on_antecedent"]


# ---------------------------------------------------------------- dilution.csv

@pytest.fixture(scope="module")
def dilution():
    path = ART / "dilution.csv"
    assert path.exists(), f"нет файла {path.relative_to(ROOT)}, запустите python lab02/run.py --fast"
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def test_dilution_rows(dilution):
    assert len(dilution) >= 3
    for col in ("context_len", "instruction_len", "share_on_instruction", "sink_share", "uniform_share"):
        assert col in dilution[0], f"в dilution.csv нет колонки {col}"
    lens = [r["context_len"] for r in dilution]
    assert all(a < b for a, b in zip(lens, lens[1:])), "context_len должна строго расти"


def test_dilution_uniform_share(dilution):
    for r in dilution:
        assert r["uniform_share"] == pytest.approx(r["instruction_len"] / r["context_len"], abs=1e-3)


def test_dilution_share_falls(dilution):
    shares = [r["share_on_instruction"] for r in dilution]
    assert all(a > b for a, b in zip(shares, shares[1:])), f"доля на инструкцию не убывает: {shares}"
