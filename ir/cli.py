"""Command line interface.

    python -m ir.cli search "effect of sweep angle on boundary layer transition"
    python -m ir.cli search "shock wave" --model lsa -k 5
    python -m ir.cli evaluate --model bm25
    python -m ir.cli evaluate --all
    python -m ir.cli explain 1        # why did query 1 rank what it ranked?
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from .data import Cranfield
from .evaluate import compare, evaluate
from .index import InvertedIndex
from .models import BM25, LatentSemanticModel, RocchioFeedback, VectorSpaceModel
from .text import DEFAULT

# Hyperparameters as selected on the development split by
# ``experiments/run_all.py`` -- not library defaults, and not retuned here.
BM25_K1, BM25_B, LSA_K = 1.5, 0.5, 150

MODELS = {
    "tfidf": lambda idx: VectorSpaceModel(idx),
    "bm25": lambda idx: BM25(idx, k1=BM25_K1, b=BM25_B),
    "lsa": lambda idx: LatentSemanticModel(idx, n_components=LSA_K),
    "prf": lambda idx: RocchioFeedback(BM25(idx, k1=BM25_K1, b=BM25_B)),
}


def _load(data_dir: str) -> tuple[Cranfield, InvertedIndex]:
    corpus = Cranfield.load(data_dir)
    index = InvertedIndex.build([d.doc_id for d in corpus.documents], corpus.texts, DEFAULT)
    return corpus, index


def cmd_search(args: argparse.Namespace) -> int:
    corpus, index = _load(args.data)
    retriever = MODELS[args.model](index)
    by_id = {d.doc_id: d for d in corpus.documents}
    for rank, (doc_id, score) in enumerate(retriever.search(args.query, k=args.k), 1):
        doc = by_id[doc_id]
        print(f"{rank:2d}. [{score:.4f}] #{doc_id}  {doc.title.strip()}")
        if args.verbose:
            print(textwrap.indent(textwrap.fill(doc.body.strip()[:300], 88), "      "))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    corpus, index = _load(args.data)
    names = list(MODELS) if args.all else [args.model]
    results = [evaluate(MODELS[n](index), corpus.queries, corpus.qrels) for n in names]

    metrics = list(results[0].per_query)
    width = max(len(r.model) for r in results) + 2
    print("model".ljust(width) + "".join(m.rjust(10) for m in metrics))
    for r in results:
        print(r.model.ljust(width) + "".join(f"{r.means[m]:10.4f}" for m in metrics))

    if len(results) > 1:
        print(f"\npaired bootstrap on {args.metric}, baseline = tfidf")
        for model, mean, delta, p in compare(results, args.metric, baseline="tfidf"):
            if p != p:                       # NaN marks the baseline row itself
                print(f"  {model:<10} {mean:.4f}  {delta:+.4f}  (baseline)")
            else:
                star = "  *" if p < 0.05 else ""
                print(f"  {model:<10} {mean:.4f}  {delta:+.4f}  p={p:.4f}{star}")
        print("  note: all 225 queries, so these include the queries the "
              "hyperparameters were tuned on;\n        the held-out numbers are in "
              "the README and in experiments/run_all.py")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Show the query analysis and the term-level evidence for the top hit."""
    corpus, index = _load(args.data)
    query = corpus.queries[args.query_id - 1]
    retriever = MODELS[args.model](index)
    tokens = retriever.analyse(query.text)

    print(f"query {query.query_id}: {query.text.strip()}")
    print(f"analysed  : {tokens}")
    print(f"relevant  : {sorted(corpus.qrels.relevant(query.query_id))}")
    hits = retriever.search(query.text, k=args.k)
    print(f"\n{'rank':<5}{'doc':<7}{'score':<10}{'judged':<9}title")
    for rank, (doc_id, score) in enumerate(hits, 1):
        grade = corpus.qrels.grades.get(query.query_id, {}).get(doc_id)
        judged = f"grade {grade}" if grade else "--"
        title = corpus.documents[doc_id - 1].title.strip().replace("\n", " ")[:52]
        print(f"{rank:<5}{doc_id:<7}{score:<10.4f}{judged:<9}{title}")

    print("\nterm evidence (df = documents containing the term):")
    for t in dict.fromkeys(tokens):
        print(f"  {t:<20} df={index.document_frequency(t):<5} idf={index.idf(t):.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ir", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data", help="directory holding the Cranfield files")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="rank documents for a free-text query")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--model", choices=MODELS, default="bm25")
    s.add_argument("-v", "--verbose", action="store_true", help="print abstracts")
    s.set_defaults(func=cmd_search)

    e = sub.add_parser("evaluate", help="score a model on the 225 Cranfield queries")
    e.add_argument("--model", choices=MODELS, default="bm25")
    e.add_argument("--all", action="store_true", help="evaluate every model and compare")
    e.add_argument("--metric", default="nDCG@10")
    e.set_defaults(func=cmd_evaluate)

    x = sub.add_parser("explain", help="show why a query ranked what it ranked")
    x.add_argument("query_id", type=int, help="1-based query number (1..225)")
    x.add_argument("-k", type=int, default=10)
    x.add_argument("--model", choices=MODELS, default="bm25")
    x.set_defaults(func=cmd_explain)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
