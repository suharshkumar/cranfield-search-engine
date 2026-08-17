# Cranfield Search Engine

A search engine built from scratch on the Cranfield 1400 collection: inverted
index, four retrieval models, and an evaluation harness that checks whether the
differences between them are real.

CS6370 Natural Language Processing, IIT Madras.

```bash
scripts/fetch_data.sh                                  # 1400 abstracts, 225 queries
pip install -r requirements.txt
python -m ir.cli search "effect of sweep angle on boundary layer transition"
python -m ir.cli evaluate --all
python -m experiments.run_all                          # regenerates everything below
```

## Results

Four models. Hyperparameters picked on 113 development queries, scores reported
on the 112 held-out ones.

| model | P@10 | R@10 | nDCG@10 | MAP | ms/query | p vs TF-IDF |
|---|---|---|---|---|---|---|
| LSA (k=150) | 0.245 | 0.426 | 0.382 | 0.313 | 0.9 | 0.093 |
| BM25 (k1=1.5, b=0.5) | 0.229 | 0.388 | 0.365 | 0.288 | 0.6 | 0.505 |
| TF-IDF cosine | 0.230 | 0.403 | 0.357 | 0.277 | 0.6 | — |
| BM25 + pseudo-relevance feedback | 0.238 | 0.404 | 0.358 | 0.299 | 2.0 | 0.961 |

LSA comes out on top. The more useful finding is that none of these gaps are
statistically significant.

A paired bootstrap over the per-query scores puts LSA's +0.025 nDCG@10 at
p = 0.09, and BM25's +0.008 over plain TF-IDF at p = 0.51. Inverting the
sample-size formula on the observed variance says you'd need 311 queries to
resolve the LSA gap at 80% power, and 1987 for the BM25 one. Cranfield has 225.
Per-query nDCG@10 has a standard deviation around 0.25, which is ten to thirty
times the size of the gaps being compared, so the collection simply can't see
them.

That changes what you can honestly say about this experiment. All four models
are within measurement error of each other. LSA is the best guess if you have
to pick one. Pseudo-relevance feedback is the only thing here that looks
actively harmful, and it costs 3x the latency to be worse.

![model comparison](results/models.png)

Looking at it per query makes the problem obvious. Switching from TF-IDF to
BM25 helps 50 queries, hurts 37, and leaves 25 alone. The +0.008 average is two
large opposing effects nearly cancelling, not a consistent improvement:

![per-query deltas](results/per_query_delta.png)

## Preprocessing matters more than the model does

Since the model choice turned out to be unmeasurable, the preprocessing
ablation is the more interesting experiment. BM25 held fixed, one stage changed
at a time, development queries:

| pipeline | vocabulary | nDCG@10 | MAP | p vs baseline |
|---|---|---|---|---|
| naive segmentation + naive tokenisation, no normalisation | 8474 | 0.338 | 0.275 | — |
| + Punkt sentence segmentation | 8474 | 0.338 | 0.275 | 1.000 |
| + Penn Treebank tokenisation | 8299 | 0.334 | 0.274 | 0.540 |
| + Porter stemming | 5672 | 0.355 | 0.299 | 0.295 |
| + WordNet lemmatisation (instead of stemming) | 6481 | 0.338 | 0.285 | 0.966 |
| **+ Porter stemming + NLTK stopwords** | **5566** | **0.361** | **0.311** | 0.166 |
| + Porter stemming + top-50 collection-frequency stopwords | 5636 | 0.339 | 0.281 | 0.939 |

Four things here surprised me:

**Punkt segmentation changes nothing at all.** Identical vocabulary, identical
scores. Punkt only knows the abbreviations in its trained model, so it handles
"Dr." and misses "fig." and "no.", which are the ones Cranfield's prose
actually uses. And sentence boundaries turn out not to matter anyway, because
the retrieval model is a bag of words that puts them straight back together.

**Treebank tokenisation makes things slightly worse.** It splits hyphenated
compounds that are single technical terms here, like "boundary-layer". Those
are exactly the rare high-IDF terms carrying the signal, so fragmenting them
costs you.

**Stemming beats lemmatisation, and not by a little.** Porter cuts the
vocabulary by a third and gains 0.017 nDCG@10. WordNet lemmatisation cuts it by
a fifth and gains nothing. Lemmatisation is the linguistically correct
operation and the wrong one for retrieval: it preserves the distinction between
"flow" and "flowing" that a searcher never intended. Porter's ugly,
ungrammatical conflation ("boundary" becomes "boundari") is what matching
actually wants.

**A generic stopword list beats one built from the collection.** The
data-driven list drops the 50 most frequent terms, which in a single-domain
aerodynamics corpus means dropping *flow*, *pressure*, *boundary*. Those are
frequent but not uninformative, because the queries are about them too. IDF
already discounts them properly; deleting them throws away real signal.

The winning pipeline is what everything downstream uses (`ir.text.DEFAULT`).

## LSA's rank matters more than the model choice

Sweeping the truncation rank moves nDCG@10 between 0.30 and 0.38. That range is
four times wider than the spread between all four models:

![LSA sweep](results/lsa_sweep.png)

Too few dimensions and separate topics get collapsed together. Too many and the
trailing singular vectors put back the term-level noise you truncated to
remove. The peak is around k = 150, which is 2.7% of the 5566-term vocabulary.

This is also why the whole thing uses a query split, and the effect of skipping
it isn't subtle. Run `python -m ir.cli evaluate --all`, which scores all 225
queries with the tuned k = 150 including the 113 that chose it, and LSA's lead
over TF-IDF grows from +0.025 to +0.033 while the p-value drops from 0.093 to
0.0017. Same code, same models, null result turned significant, purely by
reporting on the queries the hyperparameter was fitted to.

So every hyperparameter (LSA rank, BM25 k1 and b) is chosen on the odd-numbered
queries and every number in the tables comes from the even-numbered ones.

## Where it still fails

14 of the 112 held-out queries score exactly zero nDCG@10. Nothing relevant in
the top ten at all.

It isn't out-of-vocabulary terms: only 29 distinct terms out of 2234 query
tokens miss the index, and none of them appear in the failing queries. It isn't
query length either, which correlates with nDCG@10 at -0.07.

Reading the failures, they have a shape in common. They ask about a
relationship rather than a topic:

> *"does transition in the hypersonic wake depend on body geometry and size?"*
> *"can the procedure of matching inner and outer solutions be justified?"*

Every content word in those is common in the collection. Bag-of-words retrieval
finds documents about hypersonic wakes, and documents about body geometry. What
the query wants is the document about the dependence *between* them, and term
matching has no way to represent "depends on". That's a structural mismatch
rather than a vocabulary one, and none of stemming, LSA or feedback touches it.

## Notes on the implementation

**The collection has three traps.** Each one silently corrupts every score
without raising an error. All three are handled in `ir/data.py` and pinned by
tests:

1. Query ids in `cran.qry` are not the query ids in `cranqrel`. The `.I` labels
   run 1, 2, 4, ... 365 with gaps, while `cranqrel` numbers the same queries
   1-225 in sequence. Joining on `.I` agrees for the first two queries and
   mispairs every judgement after that. Queries get keyed by position instead.
2. The relevance grades are inverted. Cleverdon's scale runs 1 = "complete
   answer" down to 4 = "minimum interest", so feeding the raw grade to nDCG as
   a gain rewards the worst documents most. Grade g maps to gain 5 - g.
3. Every query carries exactly one spurious `-1` judgement. That's 225 of the
   file's 1837 rows, spread over 128 different documents. Keep them and every
   query gains a phantom relevant document.

**Everything reads one index.** `InvertedIndex` holds term -> {doc -> tf} plus a
forward index and the collection statistics. TF-IDF, BM25 and LSA are three
ways of scoring the same postings. Scoring only walks the postings lists of the
query's own terms, which is the whole point of an inverted index: under 2ms per
query over 1400 documents, with no vectorisation anywhere.

**Queries go through the document pipeline.** A stemmed index with an unstemmed
query returns nothing and raises nothing. There's a test for it.

**nDCG uses linear gain, not 2^g - 1.** Cranfield's grades are a 4-point
judgement scale, not a utility. The exponential form would claim a grade-1
document is worth 15 grade-4 ones.

**Ties break by document id** so runs reproduce across machines.

## Layout

```
ir/
  data.py       loading the collection, and the three traps above
  text.py       segment -> tokenise -> normalise -> filter
  index.py      inverted index, forward index, collection statistics
  models.py     TF-IDF cosine, BM25, LSA, Rocchio feedback
  evaluate.py   P/R/F@k, MAP, nDCG, paired bootstrap, power analysis
  cli.py        search / evaluate / explain
experiments/
  run_all.py    the four experiments; regenerates results.json and the figures
tests/          53 tests
```

`python -m ir.cli explain 38` prints the analysed query, the ranking, which
documents were judged relevant, and the document frequency of every query term.
That's the tool I used for the failure analysis above.

## Tests

```bash
python -m pytest tests -q          # 53 passed
```

The ones worth reading are in `tests/test_evaluate.py`. A metric that's subtly
wrong still returns plausible-looking numbers, so each is checked against a
ranking whose score can be worked out by hand: that MAP divides by the number
of relevant documents and not the number of hits, that the nDCG discount is
log2(rank+1), and that reversing a ranking lowers nDCG, which fails if you miss
the grade inversion.

## References

- Cleverdon (1967), *The Cranfield tests on index language devices*
- Robertson & Walker (1994), *Some simple effective approximations to the 2-Poisson model*
- Deerwester et al. (1990), *Indexing by latent semantic analysis*
- Järvelin & Kekäläinen (2002), *Cumulated gain-based evaluation of IR techniques*
- Smucker, Allan & Carterette (2007), *A comparison of statistical significance tests for IR evaluation*
