"""Tests for the three collection quirks documented in ``ir/data.py``.

Each of these silently corrupts every downstream score if handled wrongly, and
none of them produces an exception -- which is exactly why they are tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ir.data import Cranfield, load_qrels, load_queries

DATA = Path(__file__).resolve().parent.parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "cranqrel").exists(),
                                reason="collection not downloaded; run scripts/fetch_data.sh")


@pytest.fixture(scope="module")
def corpus() -> Cranfield:
    return Cranfield.load(DATA)


def test_collection_size(corpus: Cranfield) -> None:
    assert len(corpus.documents) == 1400
    assert len(corpus.queries) == 225


def test_query_ids_are_positional_not_the_printed_label(corpus: Cranfield) -> None:
    """The heart of quirk 1: the .I labels in cran.qry are sparse and must not
    be used to join against cranqrel."""
    labels = [int(q.label) for q in corpus.queries]
    assert labels[:3] == [1, 2, 4]              # 3 is missing from the file
    assert labels[-1] == 365                    # ...and the last label overshoots 225
    assert [q.query_id for q in corpus.queries] == list(range(1, 226))
    assert labels != [q.query_id for q in corpus.queries]


def test_every_judged_query_id_has_a_query(corpus: Cranfield) -> None:
    """If the join were wrong this would fail: cranqrel would reference ids
    outside 1..225 or leave queries unjudged."""
    judged = set(corpus.qrels.grades)
    assert judged == set(range(1, 226))


def test_spurious_negative_grades_are_dropped() -> None:
    """Quirk 3: every query carries exactly one grade -1 row, and dropping
    them must remove exactly those 225 rows and nothing else."""
    from collections import Counter

    rows = [line.split() for line in (DATA / "cranqrel").read_text().splitlines() if line.split()]
    bad = [r for r in rows if not 1 <= int(r[2]) <= 4]
    assert len(bad) == 225
    assert set(Counter(r[0] for r in bad).values()) == {1}      # one per query, no more
    assert {r[2] for r in bad} == {"-1"}                        # the only bad grade

    qrels = load_qrels(DATA / "cranqrel")
    assert all(1 <= g <= 4 for doc in qrels.grades.values() for g in doc.values())
    assert len(qrels) == len(rows) - len(bad) == 1612


def test_relevance_grades_are_inverted_for_gain(corpus: Cranfield) -> None:
    """Quirk 2: Cranfield grade 1 is the *best*, so gain must run the other way."""
    qid = next(q for q, d in corpus.qrels.grades.items() if 1 in d.values())
    best_doc = next(d for d, g in corpus.qrels.grades[qid].items() if g == 1)
    assert corpus.qrels.gain(qid, best_doc) == 4
    assert corpus.qrels.gain(qid, -999) == 0    # unjudged contributes nothing


def test_document_text_is_title_plus_abstract(corpus: Cranfield) -> None:
    doc = corpus.documents[0]
    assert doc.doc_id == 1
    assert doc.title and doc.body
    assert doc.text.startswith(doc.title)
    assert len(doc.text) > len(doc.title)


def test_queries_have_text(corpus: Cranfield) -> None:
    assert all(q.text.strip() for q in corpus.queries)
    assert load_queries(DATA / "cran.qry")[0].text.lower().startswith("what similarity laws")
