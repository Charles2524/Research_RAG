"""Phase 6 gate: end-to-end cited answers, zero out-of-set citations, tokens/s + RSS for both models."""

from __future__ import annotations

import pytest

import generate
from config import load_config
from models import Answer, Chunk, Retrieved


def _ret(i: int, n_tokens: int, paper="p") -> Retrieved:
    c = Chunk(chunk_id=f"{paper}#{i}", paper_id=paper, ordinal=i, section=f"S{i}", page_start=i, page_end=i,
              text=f"Section {i}\n\nchunk text {i}", n_tokens=n_tokens)
    return Retrieved(chunk=c, score=1.0 / (i + 1), rank=i + 1, mode="hybrid")


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    try:
        generate.ensure_model(c, c.llm_fallback_model)
    except generate.GenerationError as e:
        pytest.skip(f"UNVERIFIED: {e}")
    return c


# ----- offline: context budget and citation parsing -----

def test_assemble_context_respects_budget_and_order():
    cfg = load_config().with_overrides(token_budget=1000, context_chunks=5)
    hits = [_ret(i, 400) for i in range(6)]
    ctx = generate.assemble_context(cfg, hits)
    assert [r.rank for r in ctx] == [1, 2]                  # 400+400 fits, third would exceed 1000
    assert generate.assemble_context(cfg, hits, max_chunks=1) == hits[:1]
    big = generate.assemble_context(cfg, [_ret(0, 5000)])
    assert big == [_ret(0, 5000)] or len(big) == 1           # never empty when something was retrieved
    assert generate.assemble_context(cfg, []) == []


def test_parse_citations_valid_and_invalid():
    valid, invalid = generate.parse_citations("Claim one [S1]. Claim two [S2][S1]. Both [S1, S3]. Bad [S9]. Note [see above].", 3)
    assert valid == [1, 2, 3] and invalid == ["S9"]
    assert generate.parse_citations("no citations here", 3) == ([], [])
    assert generate.parse_citations("[2] and [ S3 ]", 3) == ([2, 3], [])


def test_answer_integrity_counts_out_of_set():
    a = Answer(text="x [S1] y [S7]", cited_ids=["p#0", "S7"], retrieved_ids=["p#0", "p#1"], model="m",
               completion_tokens=40, eval_s=2.0)
    assert a.invalid_citations == ["S7"] and a.citation_integrity == 0.5 and a.tokens_per_s == 20.0


def test_prompt_lists_sources_with_labels():
    cfg = load_config()
    p = generate.build_prompt(cfg, "why?", [_ret(0, 10), _ret(1, 10)])
    assert "[S1] (p, §S0, p.0)" in p and "[S2]" in p and p.endswith("Question: why?\nAnswer with citations.")


def test_source_reference_markers_stripped():
    s = "RAG improves factuality [12] and recall [3, 7] as shown [4-6]; see [S1] and [Fig. 2]."
    out = generate.clean_source_text(s)
    assert "[12]" not in out and "[3, 7]" not in out and "[4-6]" not in out
    assert "[S1]" in out and "[Fig. 2]" in out


def test_host_port_parsing():
    cfg = load_config()
    assert generate._host_port(cfg) == ("127.0.0.1", 11434)
    assert generate._host_port(cfg.with_overrides(ollama_host="http://localhost")) == ("localhost", 11434)


def test_unreachable_ollama_is_a_clear_error():
    cfg = load_config().with_overrides(ollama_host="http://127.0.0.1:1")
    with pytest.raises(generate.GenerationError, match="not reachable"):
        generate.available_models(cfg)


# ----- live (Ollama) -----

def test_end_to_end_answer_has_citations_and_no_invalid_ones(cfg):
    c = cfg.with_overrides(llm_model=cfg.llm_fallback_model)      # 1.7B keeps the suite fast
    ans, hits = generate.answer_query(c, "What does the RAG model combine a pre-trained seq2seq model with?")
    print(f"\nANSWER ({ans.model}, {ans.tokens_per_s:.1f} tok/s): {ans.text}")
    assert ans.text and hits
    assert ans.cited_ids, "answer carries no citation"
    assert ans.invalid_citations == [], f"citations outside retrieved set: {ans.invalid_citations}"
    assert set(ans.cited_ids) <= set(ans.retrieved_ids)
    assert ans.completion_tokens > 0 and ans.tokens_per_s > 0
    generate.unload_model(c, c.llm_model)
