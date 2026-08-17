"""Loading the Cranfield 1400 test collection.

The collection ships as three files in a 1960s-era tagged format:

    cran.all.1400   1400 aerodynamics abstracts   (.I .T .A .B .W)
    cran.qry         225 information needs        (.I .W)
    cranqrel        1836 relevance judgements     (qid docid grade)

Three details in here are easy to get wrong and each one silently corrupts
every score downstream.  They are handled explicitly and each is covered by a
test in ``tests/test_data.py``.

1.  **The query ids in ``cran.qry`` are not the query ids in ``cranqrel``.**
    ``cran.qry`` carries ``.I`` values that run 001, 002, 004, ... up to 365
    with gaps, while ``cranqrel`` numbers the same queries sequentially 1..225.
    Joining on the ``.I`` value looks like it works -- the two agree for the
    first two queries -- and then quietly mis-pairs every judgement after that.
    We key queries by their *ordinal position* and keep the printed ``.I`` only
    as a label.

2.  **Cranfield relevance grades are inverted.**  Cleverdon's scale runs
    1 = "a complete answer to the question" down to 4 = "minimum interest".
    Feeding the raw grade to nDCG as a gain rewards the worst documents most.
    :meth:`Qrels.gain` maps grade g to 5 - g, so 1 -> 4 ... 4 -> 1.

3.  **A spurious ``-1`` judgement.**  Every one of the 225 queries carries
    exactly one row with grade -1 -- 225 of the file's 1837 rows, 12% of it,
    spread over 128 distinct documents.  ``-1`` is outside Cleverdon's 1-4
    scale and is not a relevance grade, so those rows are dropped; keeping
    them would add a phantom relevant document to every single query.
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
