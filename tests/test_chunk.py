"""Phase 2: chunking. Synthetic markdown here; the real-corpus checks live in test_parse.py."""

from __future__ import annotations

import pytest

import chunk as chunk_mod
from chunk import chunk_markdown, count_tokens, describe, split_sections
from config import load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    try:
        chunk_mod.get_tokenizer(c)
    except chunk_mod.TokenizerMissing as e:
        pytest.skip(str(e))
    return c


def _sentences(n: int, tag: str = "s") -> str:
    return " ".join(f"This is {tag} sentence number {i} about dense passage retrieval and rerankers." for i in range(n))


MD = f"""<!-- page:1 -->
# 1 Introduction

{_sentences(5, 'intro')}

<!-- page:2 -->
{_sentences(5, 'intro2')}

## 2.1 Dense Retrieval

{_sentences(60, 'dense')}

<!-- page:3 -->
{_sentences(60, 'dense3')}

# 3 Results

Short section.
"""


def test_split_sections_tracks_titles_and_pages():
    secs = split_sections(MD)
    assert [s.title for s in secs] == ["1 Introduction", "2.1 Dense Retrieval", "3 Results"]
    assert (secs[0].page_start, secs[0].page_end) == (1, 2)
    assert (secs[1].page_start, secs[1].page_end) == (2, 3)
    assert (secs[2].page_start, secs[2].page_end) == (3, 3)
    assert "Short section." in secs[2].text and "#" not in secs[2].text


def test_chunks_carry_ids_sections_pages_and_prefix(cfg):
    chunks = chunk_markdown(cfg, "p1", MD, size_tokens=256, overlap_tokens=32)
    assert chunks and [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.chunk_id == f"p1#{c.ordinal}" and c.paper_id == "p1"
        assert c.section and c.page_start >= 1 and c.page_end >= c.page_start
        assert c.text.startswith(c.section + "\n\n")
        assert c.text.strip() and c.n_tokens == count_tokens(cfg, c.text)
        assert c.n_tokens <= 256
    assert {c.section for c in chunks} == {"1 Introduction", "2.1 Dense Retrieval", "3 Results"}
    dense = [c for c in chunks if c.section == "2.1 Dense Retrieval"]
    assert len(dense) >= 4                                   # ~1500 tokens of body split at 256
    assert dense[0].page_start == 2 and dense[-1].page_end == 3


def test_overlap_repeats_text_between_neighbours(cfg):
    chunks = [c for c in chunk_markdown(cfg, "p", MD, 256, 48) if c.section == "2.1 Dense Retrieval"]
    a, b = chunks[0].text, chunks[1].text
    body_a = a.split("\n\n", 1)[1]
    tail = body_a[-60:]
    assert tail.split()[1] in b                              # a word from the tail of a appears in b


def test_no_overlap_when_zero(cfg):
    chunks = [c for c in chunk_markdown(cfg, "p", MD, 128, 0) if c.section == "2.1 Dense Retrieval"]
    bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
    joined = "".join(bodies).replace(" ", "").replace("\n", "")
    original = "".join(s.text for s in split_sections(MD) if s.title == "2.1 Dense Retrieval").replace(" ", "").replace("\n", "")
    assert joined == original                                # zero overlap => bodies tile the section exactly


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_size_cap_respected_for_every_size(cfg, size):
    big = "# Long\n\n" + _sentences(400)
    chunks = chunk_markdown(cfg, "p", big, size, 64)
    assert chunks and max(c.n_tokens for c in chunks) <= size
    assert all(c.n_tokens > 0 for c in chunks)


def test_tiny_trailing_fragment_is_merged(cfg):
    md = "# S\n\n" + _sentences(30)
    chunks = chunk_markdown(cfg, "p", md, 512, 0)
    assert len(chunks) == 1


def test_empty_and_heading_only_markdown(cfg):
    assert chunk_markdown(cfg, "p", "", 256, 0) == []
    assert chunk_markdown(cfg, "p", "# Only a heading\n\n## Another\n", 256, 0) == []


def test_describe():
    assert describe([]) == {"count": 0}
    chunks = chunk_markdown(load_config(), "p", MD, 256, 32)
    d = describe(chunks)
    assert d["count"] == len(chunks) and d["min"] <= d["p50"] <= d["max"] <= 256
