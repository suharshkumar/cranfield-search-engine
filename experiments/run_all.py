"""Reproduce every number in the README.

    python -m experiments.run_all            # ~4 minutes on a laptop

Writes ``results/results.json`` and the figures in ``results/``.  The four
experiments answer four questions, in order:

    1. Does better linguistics make better retrieval?     (preprocessing)
    2. How many latent dimensions does LSA want, and      (tuning, on dev only)
       which BM25 parameters?
    3. Which retrieval model wins, and is the win real?   (held-out + bootstrap)
    4. Where does the best system still fail?             (error analysis)

Every hyperparameter is chosen on the 113 odd-numbered queries and every
reported number comes from the 112 even-numbered ones.  This matters more than
it sounds: LSA's dimensionality swings nDCG@10 by 0.07 across the sweep, so
picking ``k`` on the queries you then report on manufactures most of a result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ir.data import Cranfield
from ir.evaluate import compare, evaluate, paired_bootstrap, queries_needed
from ir.index import InvertedIndex
from ir.models import BM25, LatentSemanticModel, RocchioFeedback, VectorSpaceModel
from ir.text import DEFAULT, Preprocessor, frequency_stopwords, nltk_stopwords

RESULTS = Path(__file__).resolve().parent.parent / "results"
METRIC = "nDCG@10"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- 1


def experiment_preprocessing(corpus: Cranfield) -> dict:
    """Ablate one pipeline stage at a time, holding the retrieval model fixed.

    BM25 is the fixed model because it has no learned parameters, so any
    difference is attributable to the pipeline and not to refitting.
    """
    _log("experiment 1: preprocessing ablation")

    # A stopword list derived from the collection itself needs the tokens first.
    tokens = Preprocessor(normaliser="none", stopwords=None).batch(corpus.texts)
    freq_stops = frequency_stopwords(tokens, top_n=50)

    configs = [
        ("baseline (naive everything)", Preprocessor("naive", "naive", "none", None)),
        ("+ Punkt segmentation",        Preprocessor("punkt", "naive", "none", None)),
        ("+ Treebank tokenisation",     Preprocessor("punkt", "treebank", "none", None)),
        ("+ Porter stemming",           Preprocessor("punkt", "treebank", "stem", None)),
        ("+ WordNet lemmatisation",     Preprocessor("punkt", "treebank", "lemma", None)),
        ("stem + NLTK stopwords",       Preprocessor("punkt", "treebank", "stem", nltk_stopwords())),
        ("stem + top-50 frequency stops", Preprocessor("punkt", "treebank", "stem", freq_stops)),
    ]

    out, baseline = [], None
    for label, pre in configs:
        t0 = time.time()
        index = InvertedIndex.build([d.doc_id for d in corpus.documents], corpus.texts, pre)
        res = evaluate(BM25(index), corpus.queries, corpus.qrels)
        if baseline is None:
            baseline = res
        p = paired_bootstrap(res[METRIC], baseline[METRIC]) if res is not baseline else None
        out.append({"label": label, "config": pre.name, "vocab": len(index.vocabulary),
                    "build_s": round(time.time() - t0, 1),
                    "p_vs_baseline": None if p is None else round(p, 4),
                    "queries_needed": (None if res is baseline else
                                       queries_needed(res[METRIC], baseline[METRIC])),
                    **res.means})
        _log(f"    {label:32s} |V|={len(index.vocabulary):6d}  "
             f"{METRIC}={res.means[METRIC]:.4f}" + ("" if p is None else f"  p={p:.3f}"))
    return {"configs": out, "frequency_stopwords": sorted(freq_stops)}


# ---------------------------------------------------------------------- 2


def experiment_tuning(index: InvertedIndex, dev) -> dict:
    """Sweep LSA's rank and BM25's (k1, b) -- on the development queries only."""
    _log("experiment 2: hyperparameter search (development queries only)")

    lsa_rows = []
    for k in (50, 100, 150, 200, 250, 300, 400, 500, 800):
        r = evaluate(LatentSemanticModel(index, n_components=k), dev.queries, dev.qrels)
        lsa_rows.append({"k": k, **{m: round(v, 4) for m, v in r.means.items()}})
        _log(f"    LSA  k={k:4d}  dev {METRIC}={r.means[METRIC]:.4f}")
    best_lsa = max(lsa_rows, key=lambda r: r[METRIC])

    bm_rows = []
    for k1 in (0.9, 1.2, 1.5, 2.0):
        for b in (0.0, 0.3, 0.5, 0.75, 1.0):
            r = evaluate(BM25(index, k1=k1, b=b), dev.queries, dev.qrels)
            bm_rows.append({"k1": k1, "b": b, METRIC: round(r.means[METRIC], 4),
                            "MAP": round(r.means["MAP"], 4)})
    best_bm = max(bm_rows, key=lambda r: r[METRIC])
    _log(f"    BM25 best k1={best_bm['k1']} b={best_bm['b']}  "
         f"dev {METRIC}={best_bm[METRIC]:.4f}")

    return {"lsa_sweep": lsa_rows, "lsa_best": best_lsa,
            "bm25_grid": bm_rows, "bm25_best": best_bm}


# ---------------------------------------------------------------------- 3


def experiment_models(index: InvertedIndex, test, tuning: dict) -> dict:
    """Score every model on the held-out queries with the dev-chosen settings."""
    _log("experiment 3: model comparison (held-out queries)")
    k1, b = tuning["bm25_best"]["k1"], tuning["bm25_best"]["b"]
    lsa_k = tuning["lsa_best"]["k"]

    bm25 = BM25(index, k1=k1, b=b)
    models = [VectorSpaceModel(index), bm25,
              LatentSemanticModel(index, n_components=lsa_k), RocchioFeedback(bm25)]

    results, latency = [], {}
    for m in models:
        t0 = time.time()
        r = evaluate(m, test.queries, test.qrels)
        latency[m.name] = round((time.time() - t0) / len(test.queries) * 1000, 2)
        results.append(r)
        _log(f"    {m.name:12s} {METRIC}={r.means[METRIC]:.4f}  MAP={r.means['MAP']:.4f}  "
             f"{latency[m.name]:.1f} ms/query")

    table = compare(results, METRIC, baseline="tfidf")
    return {
        "settings": {"bm25_k1": k1, "bm25_b": b, "lsa_k": lsa_k},
        "means": {r.model: r.means for r in results},
        "latency_ms": latency,
        "vs_tfidf": [{"model": m, METRIC: round(v, 4), "delta": round(d, 4),
                      "p": None if np.isnan(p) else round(p, 4)} for m, v, d, p in table],
        "per_query": {r.model: r[METRIC].tolist() for r in results},
    }


# ---------------------------------------------------------------------- 4


def experiment_errors(index: InvertedIndex, corpus, models: dict) -> dict:
    """Which queries does the best system fail, and do they share a cause?"""
    _log("experiment 4: failure analysis")
    scores = np.asarray(models["per_query"]["bm25"])
    order = np.argsort(scores)
    worst = []
    for i in order[:10]:
        q = corpus.queries[int(i)]
        tokens = index.preprocessor(q.text)
        oov = [t for t in tokens if t not in index.postings]
        rare = sorted(((index.document_frequency(t), t) for t in tokens if t in index.postings))[:3]
        worst.append({
            "query_id": q.query_id, "text": q.text.strip()[:120],
            "nDCG@10": round(float(scores[i]), 4),
            "n_relevant": len(corpus.qrels.relevant(q.query_id)),
            "n_terms": len(tokens),
            "out_of_vocabulary": oov,
            "rarest_terms": [{"term": t, "df": df} for df, t in rare],
        })
    lengths = np.asarray([len(index.preprocessor(q.text)) for q in corpus.queries])
    n_rel = np.asarray([len(corpus.qrels.relevant(q.query_id)) for q in corpus.queries])
    return {
        "worst_queries": worst,
        "corr_length_ndcg": round(float(np.corrcoef(lengths, scores)[0, 1]), 3),
        "corr_nrelevant_ndcg": round(float(np.corrcoef(n_rel, scores)[0, 1]), 3),
        "n_queries_zero_ndcg": int((scores == 0).sum()),
    }


# ---------------------------------------------------------------------- plots


def make_figures(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(exist_ok=True)

    # Precision-recall style: metric @ k for each model
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ks = [1, 5, 10]
    for model, means in report["models"]["means"].items():
        ax[0].plot(ks, [means[f"nDCG@{k}"] for k in ks], marker="o", label=model)
        ax[1].plot([means[f"R@{k}"] for k in ks], [means[f"P@{k}"] for k in ks],
                   marker="o", label=model)
    ax[0].set(xlabel="k", ylabel="nDCG@k", title="Ranking quality by cutoff", xticks=ks)
    ax[1].set(xlabel="Recall@k", ylabel="Precision@k", title="Precision against recall")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "models.png", dpi=140)
    plt.close(fig)

    # LSA sweep (development queries -- this is the curve k was chosen from)
    sweep = report["tuning"]["lsa_sweep"]
    fig, a = plt.subplots(figsize=(5.5, 4))
    a.plot([r["k"] for r in sweep], [r["nDCG@10"] for r in sweep], marker="o", label="LSA (dev)")
    a.axvline(report["models"]["settings"]["lsa_k"], ls="--", c="#c0392b", lw=1,
              label=f"chosen k={report['models']['settings']['lsa_k']}")
    a.set(xlabel="latent dimensions k", ylabel="nDCG@10 (development queries)",
          title="LSA is sharply sensitive to its rank")
    a.legend(fontsize=8)
    a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "lsa_sweep.png", dpi=140)
    plt.close(fig)

    # Per-query deltas: BM25 vs TF-IDF, sorted -- shows how few queries drive the mean
    a_s = np.asarray(report["models"]["per_query"]["bm25"])
    b_s = np.asarray(report["models"]["per_query"]["tfidf"])
    d = np.sort(a_s - b_s)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(d)), d, color=["#c0392b" if x < 0 else "#27ae60" for x in d], width=1.0)
    ax.axhline(0, c="k", lw=.8)
    ax.set(xlabel="query (sorted by change)", ylabel="nDCG@10  BM25 - TF-IDF",
           title="Per-query effect of switching to BM25")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "per_query_delta.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------- main


def split(corpus: Cranfield) -> tuple[Cranfield, Cranfield]:
    """Odd-numbered queries tune, even-numbered queries report.

    A deterministic interleave rather than a random split: it needs no seed to
    reproduce, and it cannot accidentally put all the easy queries on one side
    the way a small random draw can.
    """
    dev = Cranfield(corpus.documents, corpus.queries[0::2], corpus.qrels)
    test = Cranfield(corpus.documents, corpus.queries[1::2], corpus.qrels)
    return dev, test


def main() -> None:
    corpus = Cranfield.load()
    _log(f"loaded {len(corpus.documents)} documents, {len(corpus.queries)} queries, "
         f"{len(corpus.qrels)} judgements")
    dev, test = split(corpus)
    _log(f"split: {len(dev.queries)} development / {len(test.queries)} held-out queries")

    report: dict = {"split": {"dev": len(dev.queries), "test": len(test.queries)}}
    report["preprocessing"] = experiment_preprocessing(dev)

    # The index every later experiment uses is built with the ablation's
    # winner (ir.text.DEFAULT), so the pipeline choice is carried through
    # rather than quietly reset to something else.
    index = InvertedIndex.build([d.doc_id for d in corpus.documents], corpus.texts, DEFAULT)
    report["index"] = {"vocabulary": len(index.vocabulary), "n_docs": index.n_docs,
                       "avg_doc_length": round(index.avg_doc_length, 1),
                       "postings": sum(len(p) for p in index.postings.values())}

    report["tuning"] = experiment_tuning(index, dev)
    report["models"] = experiment_models(index, test, report["tuning"])
    report["errors"] = experiment_errors(index, test, report["models"])

    # Headline claims, stated as tests rather than as assertions.
    per_query = report["models"]["per_query"]
    bm, tf, lsa, prf = (np.asarray(per_query[m])
                        for m in ("bm25", "tfidf", "lsa", "bm25+prf"))
    report["headline"] = {
        name: {"delta": round(float(x.mean() - y.mean()), 4),
               "p": round(paired_bootstrap(x, y), 5),
               "queries_needed_for_significance": queries_needed(x, y)}
        for name, (x, y) in {
            "bm25_vs_tfidf": (bm, tf), "lsa_vs_tfidf": (lsa, tf),
            "lsa_vs_bm25": (lsa, bm), "prf_vs_bm25": (prf, bm),
        }.items()
    }
    report["headline"]["n_test_queries"] = len(test.queries)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2))
    make_figures(report)
    _log(f"wrote {RESULTS/'results.json'} and 3 figures")


if __name__ == "__main__":
    main()
