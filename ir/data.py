"""Loading the Cranfield 1400 collection.

Three files, in a 1960s tagged format:

    cran.all.1400   1400 aerodynamics abstracts   (.I .T .A .B .W)
    cran.qry         225 information needs        (.I .W)
    cranqrel        1837 relevance judgements     (qid docid grade)

Three gotchas here, all of which quietly wreck your scores rather than throwing
anything. Tests for each in tests/test_data.py.

1. The query ids in cran.qry are not the query ids in cranqrel. cran.qry's .I
   values run 001, 002, 004 ... 365 with gaps; cranqrel numbers the same
   queries 1..225 in order. They agree for the first two queries, so joining on
   .I looks fine and then mispairs everything after. Key queries by position
   and keep .I as a label only.

2. The grades are inverted. Cleverdon's scale is 1 = "complete answer" down to
   4 = "minimum interest", so passing the raw grade to nDCG as a gain rewards
   the worst documents most. Qrels.gain maps g to 5 - g.

3. Every query has exactly one row with grade -1: 225 of the 1837 rows, over
   128 different documents. It's outside the 1-4 scale and isn't a relevance
   grade, so drop those rows. Keep them and every query gains a phantom
   relevant document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Cranfield's own scale: 1 is the best grade, 4 the worst.
BEST_GRADE, WORST_GRADE = 1, 4


@dataclass(frozen=True)
class Document:
    doc_id: int          # 1-based position in cran.all.1400
    title: str
    author: str
    bibliography: str
    body: str

    @property
    def text(self) -> str:
        """Title + abstract, the retrievable content.

        The ``.W`` field of a Cranfield record repeats the title as its first
        sentence, so this is not a doubling of the title -- it is the record as
        the collection defines it.
        """
        return f"{self.title}\n{self.body}".strip()


@dataclass(frozen=True)
class Query:
    query_id: int        # 1-based ordinal -- the id cranqrel uses
    label: str           # the .I value printed in cran.qry, kept for reference
    text: str


@dataclass
class Qrels:
    """Relevance judgements, keyed by the sequential query id."""

    grades: dict[int, dict[int, int]] = field(default_factory=dict)

    def relevant(self, query_id: int) -> set[int]:
        """Documents judged relevant at any grade (binary view, for MAP/P@k)."""
        return set(self.grades.get(query_id, {}))

    def gain(self, query_id: int, doc_id: int) -> int:
        """Graded gain for nDCG, with Cranfield's inverted scale corrected.

        Grade 1 ("complete answer") -> gain 4; grade 4 ("minimum interest") ->
        gain 1; unjudged -> 0.  Cranfield is judged shallowly, so unjudged is
        treated as non-relevant, the standard TREC assumption.
        """
        grade = self.grades.get(query_id, {}).get(doc_id)
        return 0 if grade is None else (WORST_GRADE + 1) - grade

    def __len__(self) -> int:
        return sum(len(v) for v in self.grades.values())


def _split_records(raw: str) -> list[list[str]]:
    """Split a Cranfield file into records on the ``.I`` marker."""
    records, current = [], None
    for line in raw.splitlines():
        if line.startswith(".I"):
            if current is not None:
                records.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        records.append(current)
    return records


def _parse_fields(lines: list[str]) -> tuple[str, dict[str, str]]:
    """Return the ``.I`` label and a tag -> text mapping for one record."""
    label = lines[0][2:].strip()
    fields: dict[str, list[str]] = {}
    tag = None
    for line in lines[1:]:
        if re.fullmatch(r"\.[TABW]", line.strip()):
            tag = line.strip()[1]
            fields.setdefault(tag, [])
        elif tag is not None:
            fields[tag].append(line)
    return label, {k: "\n".join(v).strip() for k, v in fields.items()}


def load_documents(path: Path) -> list[Document]:
    records = _split_records(Path(path).read_text(encoding="utf-8", errors="replace"))
    docs = []
    for i, rec in enumerate(records, start=1):
        _, f = _parse_fields(rec)
        docs.append(Document(
            doc_id=i,
            title=f.get("T", ""),
            author=f.get("A", ""),
            bibliography=f.get("B", ""),
            body=f.get("W", ""),
        ))
    return docs


def load_queries(path: Path) -> list[Query]:
    """Load queries, numbering them 1..N by position -- see note 1 in the module docstring."""
    records = _split_records(Path(path).read_text(encoding="utf-8", errors="replace"))
    queries = []
    for i, rec in enumerate(records, start=1):
        label, f = _parse_fields(rec)
        queries.append(Query(query_id=i, label=label, text=f.get("W", "")))
    return queries


def load_qrels(path: Path) -> Qrels:
    """Load relevance judgements, dropping the ``-1`` artefact rows."""
    grades: dict[int, dict[int, int]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        qid, doc_id, grade = (int(p) for p in parts)
        if not BEST_GRADE <= grade <= WORST_GRADE:
            continue    # the spurious -1 row every query carries -- see note 3
        grades.setdefault(qid, {})[doc_id] = grade
    return Qrels(grades)


@dataclass
class Cranfield:
    documents: list[Document]
    queries: list[Query]
    qrels: Qrels

    @classmethod
    def load(cls, data_dir: str | Path = "data") -> "Cranfield":
        d = Path(data_dir)
        missing = [f for f in ("cran.all.1400", "cran.qry", "cranqrel") if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"missing {missing} in {d.resolve()} -- run scripts/fetch_data.sh first")
        return cls(load_documents(d / "cran.all.1400"),
                   load_queries(d / "cran.qry"),
                   load_qrels(d / "cranqrel"))

    @property
    def texts(self) -> list[str]:
        return [d.text for d in self.documents]
