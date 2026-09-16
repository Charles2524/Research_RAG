"""Dataclasses shared across the pipeline: Paper, Chunk, Retrieved, Answer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

# Fetch outcome statuses (SPEC 5.4). "pending" means not yet attempted.
STATUS_PENDING = "pending"
STATUS_FETCHED = "fetched"
STATUS_NO_OA_PDF = "no_oa_pdf"
STATUS_HTTP_ERROR = "http_error"
STATUS_TIMEOUT = "timeout"
STATUSES = (STATUS_PENDING, STATUS_FETCHED, STATUS_NO_OA_PDF, STATUS_HTTP_ERROR, STATUS_TIMEOUT)


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    openalex_id: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    venue: str | None = None
    abstract: str | None = None
    source: str = ""                 # adapter that first produced the record
    pdf_url: str | None = None
    status: str = STATUS_PENDING
    metadata_resolved: bool = True   # False => filename-based citation (SPEC 6.3)
    references: list[str] = field(default_factory=list)   # DOIs, from the excised References section

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r}; expected one of {STATUSES}")

    @property
    def has_full_text(self) -> bool:
        return self.status == STATUS_FETCHED

    def citation_label(self) -> str:
        """Short label for citations, e.g. 'Lewis 2020'. Handles 'Patrick Lewis' and 'Lewis P' forms."""
        if not self.authors:
            return f"Unknown {self.year}" if self.year else "Unknown"
        parts = [p for p in self.authors[0].replace(",", " ").split() if p]
        surname = parts[-1] if len(parts[-1].rstrip(".")) > 2 else parts[0]    # 'Lewis P' -> Lewis
        return f"{surname} {self.year}" if self.year else surname

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "Paper":
        return cls.from_dict(json.loads(s))


@dataclass
class Chunk:
    chunk_id: str        # f"{paper_id}#{ordinal}"
    paper_id: str
    ordinal: int
    section: str
    page_start: int
    page_end: int
    text: str            # section title already prepended (SPEC 6.4)
    n_tokens: int

    @staticmethod
    def make_id(paper_id: str, ordinal: int) -> str:
        return f"{paper_id}#{ordinal}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rank: int
    mode: str            # bm25 | dense | hybrid | rerank


@dataclass
class Answer:
    text: str
    cited_ids: list[str]
    retrieved_ids: list[str]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0       # wall time of the request (includes model load on first call)
    eval_s: float = 0.0          # pure generation time reported by Ollama (tokens/s denominator)
    prompt_eval_s: float = 0.0   # prompt processing time reported by Ollama (dominant on CPU)
    load_s: float = 0.0          # model load time reported by Ollama (first call only)
    thinking_chars: int = 0      # length of Ollama's separate reasoning field (its tokens are in completion_tokens)
    think: bool = False          # whether chain-of-thought was requested

    @property
    def prompt_tokens_per_s(self) -> float:
        return self.prompt_tokens / self.prompt_eval_s if self.prompt_eval_s > 0 else 0.0

    @property
    def invalid_citations(self) -> list[str]:
        """Cited IDs not present in the retrieved set (SPEC 7.3). Never silently dropped."""
        allowed = set(self.retrieved_ids)
        return [c for c in self.cited_ids if c not in allowed]

    @property
    def citation_integrity(self) -> float:
        """Fraction of citations that resolve to a retrieved chunk; 1.0 when there are none."""
        if not self.cited_ids:
            return 1.0
        return 1.0 - len(self.invalid_citations) / len(self.cited_ids)

    @property
    def tokens_per_s(self) -> float:
        denom = self.eval_s or self.latency_s
        return self.completion_tokens / denom if denom > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["invalid_citations"] = self.invalid_citations
        d["citation_integrity"] = self.citation_integrity
        d["tokens_per_s"] = self.tokens_per_s
        return d
