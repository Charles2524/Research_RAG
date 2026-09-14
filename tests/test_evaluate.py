"""Phase 5 gate: eval harness loads the eval set and produces a results row for all three modes."""

from __future__ import annotations

import csv
import json

import pytest

import evaluate
from config import RETRIEVAL_MODES, load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    if not c.eval_set_path.exists():
        pytest.skip(f"{c.eval_set_path} missing (SPEC 9.1: user-written/verified eval set)")
    return c


# ----- metric arithmetic (synthetic ranked lists; this tests the math, not retrieval) -----

def test_retrieval_metrics_math():
    ranked = [["a", "a", "b", "c"],        # relevant a at paper-rank 1
              ["x", "y", "b", "b", "b"],   # relevant b at paper-rank 3
              ["x", "y", "z"]]             # miss
    relevant = [{"a"}, {"b"}, {"q"}]
    m = evaluate.retrieval_metrics(ranked, relevant, ks=(1, 2, 5))
    assert m["recall_at_1"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recall_at_2"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recall_at_5"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-4)
    assert evaluate.retrieval_metrics([], []) == {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}


def test_chunk_contains_quote_is_whitespace_insensitive():
    assert evaluate.chunk_contains_quote("Intro\n\nThe  retriever\nuses DPR here.", "the retriever uses DPR")
    assert not evaluate.chunk_contains_quote("The retriever uses BM25.", "uses DPR")
    assert not evaluate.chunk_contains_quote("anything", "")


def test_eval_set_validation(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text('{"id":"q1","question":"?","answer":"a","paper_ids":["p"]}\n\n', encoding="utf-8")
    items = evaluate.load_eval_set(load_config(), p)
    assert len(items) == 1 and items[0]["verified"] is False
    assert "UNVERIFIED 1/1" in evaluate.eval_set_note(items)
    p.write_text('{"id":"q1","question":"?","answer":"a","paper_ids":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="paper_ids"):
        evaluate.load_eval_set(load_config(), p)
    p.write_text('{"id":"q1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        evaluate.load_eval_set(load_config(), p)
    with pytest.raises(FileNotFoundError):
        evaluate.load_eval_set(load_config(), tmp_path / "nope.jsonl")


# ----- real eval set on the real index -----

def test_eval_set_is_well_formed_against_corpus(cfg):
    import fetch
    items = evaluate.load_eval_set(cfg)
    assert 25 <= len(items) <= 40
    known = {p.paper_id for p in fetch.load_corpus(cfg)}
    for it in items:
        assert it["question"].strip() and it["answer"].strip()
        assert all(pid in known for pid in it["paper_ids"]), it["id"]
    assert len({pid for it in items for pid in it["paper_ids"]}) >= 15


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_run_eval_produces_row_per_mode(cfg, mode, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "log_run", lambda c, row, path=None: evaluate.__dict__["_orig_log"](c, row, tmp_path / "runs.csv"))
    row = evaluate.run_eval(cfg, f"t_{mode}", mode=mode, rerank=False, k=10)
    for key in ("recall_at_5", "recall_at_10", "mrr"):
        assert 0.0 <= row[key] <= 1.0
    assert row["recall_at_5"] <= row["recall_at_10"]
    assert 0.0 <= row["chunk_recall_at_10"] <= row["recall_at_10"]      # chunk-level is strictly stricter
    assert row["p50_latency_s"] > 0 and row["retrieval_mode"] == mode
    assert len(row["per_question"]) == len(evaluate.load_eval_set(cfg))
    rows = list(csv.DictReader((tmp_path / "runs.csv").open(encoding="utf-8")))
    assert rows[-1]["run_name"] == f"t_{mode}" and rows[-1]["notes"].startswith("eval_set n=")
    print(f"\nEVAL {mode}: recall@5={row['recall_at_5']} recall@10={row['recall_at_10']} mrr={row['mrr']} "
          f"p50={row['p50_latency_s']}s  [{row['notes']}]")


evaluate.__dict__.setdefault("_orig_log", evaluate.log_run)
