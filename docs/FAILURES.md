# What failed and how it was fixed

Every number here comes from a file in this repository: a report in `reports/`, the gate in
`configs/`, or an assertion in `tests/`. Retriever experiments use 40 dev contracts drawn with
seed 0 (573 answerable questions, `reports/experiments/`). The only test-split numbers are the
final BM25 results. Metrics are over answerable questions at k=10; a chunk counts as relevant if
it overlaps an expert-annotated span.

## Shipped result

BM25 over SQLite FTS5, each question searched only inside its own contract, CUAD question text
verbatim (`reports/bm25-baseline-test.json`, 102 test contracts, 1,244 answerable questions):

| recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|
| 0.5514 | 0.6706 | 0.4780 | 0.5048 |

## Summary

| Failure | Fix | Status |
|---|---|---|
| Metadata questions share no words with their answers | Cross-encoder rerank tried | Open: helps metadata, hurts overall |
| Dense retrieval below BM25 | Hybrid fusion tried | Not shipped: no meaningful gain over BM25 |
| Cross-encoder rerank below BM25 overall | None | Not shipped |
| Clauses cut at chunk boundaries | Clause-aware chunking | Mitigated, no baseline to compare |
| Quotes the contract does not contain | Grounding every quote, strict abstention | Fixed for the check, LLM rate unmeasured |
| Most questions are unanswerable | Retrieval gate plus abstention rules | Built, effect not yet reproducible |

## 1. Metadata questions share no words with their answers

On dev, BM25 finds clauses whose wording matches the question, and misses questions whose answer
is a name or a date (`reports/bm25-baseline-dev.json`, recall@10):

| Category | n | recall@10 |
|---|---|---|
| Governing Law | 354 | 0.9831 |
| Parties | 407 | 0.2271 |
| Document Name | 408 | 0.2721 |
| Agreement Date | 377 | 0.3369 |

Three alternatives on the 40-contract sample, recall@10 (n per category is small, 38 to 40):

| Category | BM25 | Dense | Hybrid | Rerank |
|---|---|---|---|---|
| Parties | 0.2000 | 0.2125 | 0.2000 | 0.3500 |
| Document Name | 0.2250 | 0.3250 | 0.2250 | 0.2500 |
| Agreement Date | 0.3158 | 0.3684 | 0.3553 | 0.7368 |

The reranker more than doubles Agreement Date recall but lowers overall recall (section 3), so it
is not shipped. Routing only the metadata categories to the reranker is the obvious next step; it
is neither built nor measured.

## 2. Dense retrieval is below BM25

All retrievers on the same 573 sampled dev questions:

| Retriever | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|
| BM25 | 0.5640 | 0.6743 | 0.5081 | 0.5279 |
| Dense, bge-small-en-v1.5 | 0.4893 | 0.5843 | 0.4358 | 0.4469 |
| Hybrid, reciprocal rank fusion | 0.5741 | 0.6770 | 0.5079 | 0.5257 |
| BM25 + cross-encoder rerank (section 3) | 0.5583 | 0.6443 | 0.4888 | 0.5030 |

Dense loses 9 points of recall@10. Fusing it with BM25 gains less than one point of recall@10 and
loses slightly on mrr@10 and ndcg@10: not worth an embedding model in the critical path.

Hypothesis, not tested: every CUAD question starts with the same long template ("Highlight the
parts (if any) of this contract related to ... that should be reviewed by a lawyer"), which
dominates the query embedding. Stripping the template is the experiment that would test it.

## 3. Cross-encoder rerank is below BM25 overall

`Xenova/ms-marco-MiniLM-L-6-v2` rescoring the top 50 BM25 chunks: recall@10 0.6443 against 0.6743,
mrr@10 0.4888 against 0.5081. The model is trained on web search passages, not contracts. Depth 50
was chosen on dev, where BM25 recall@50 bounds what reranking can reach (recorded in the commit
that set it).

## 4. Clauses cut at chunk boundaries

Fixed-size chunks can split a long clause, such as a license grant, so that no single chunk holds
the answer. Chunking cuts at blank lines and clause headings ("7.", "ARTICLE IV", "Section 3"),
falls back to sentences, and keeps exact character offsets. 13,505 expert spans lie entirely
inside one chunk (`tests/integration/test_citations_cuad_release.py`). No fixed-size baseline was
measured, so the improvement itself is not quantified.

## 5. Quotes the contract does not contain

A model asked for verbatim quotes can still paraphrase, fix typos or collapse whitespace. Every
claimed quote is located inside the chunk it cites: verbatim after whitespace and quote-mark
normalization, then ignoring case, then fuzzy (score 90 or more, quotes of 20 characters or more).
The citation shown is the contract's own text at document offsets, never the model's quote. One
quote that cannot be located withholds the whole answer.

Gold spans used as quotes (`tests/integration/test_citations_cuad_release.py`):

- verbatim and whitespace-collapsed: 13,505 of 13,505 grounded without fuzzy matching, 13,372 on
  the gold span itself, the rest on an earlier passage of the same chunk that is identical up to
  case and whitespace;
- one word shortened by a character, 9,948 quotes of 40 characters or more: all grounded, 9,921
  overlapping the gold span, the rest on a near-identical passage.

How often a real model's quotes ground is not measured.

## 6. Most questions are unanswerable

Of 4,182 test questions, 2,938 have no answer in their contract
(`reports/bm25-baseline-test.json`). Two layers withhold answers:

- before the model: per-category BM25 score thresholds (`configs/bm25-gate.json`, 37 of 41
  categories), calibrated on dev to keep 0.95 of answerable questions. Raw BM25 scores are only
  comparable within one category, because each category asks one fixed question;
- after the model: withheld when the model reports the clause absent, cites nothing, or gives a
  quote that cannot be located.

The gate's effect on unanswerable questions was measured once with a script that is not in the
repository, so it is not stated here.

## 7. Engineering failures and closed risks

- Observed: `uv sync` on Windows failed with os error 396: the Windows cloud-files driver refused
  the hard links uv installs with by default. Fixed with `UV_LINK_MODE=copy`.
- Prevented: Ollama's default context is small and it truncates longer prompts silently. `num_ctx`
  is sent on every request, and prompts that may not fit are refused before the call.
- Prevented: question text would reach the FTS5 query parser, where quotes, `NEAR` or a stray
  parenthesis break or change the query. Queries are reduced to words
  (`tests/unit/test_retrieval_bm25.py`).
- Observed: an early dense and hybrid comparison was run on the test split. Its numbers are
  discarded, `cuad-qa experiment` refuses any split but dev, and CI never reads test.

## Not measured

- End-to-end answers with a model: correctness, citation precision against gold spans, abstention
  precision and recall. Only smoke tests against a live Ollama exist.
- The retrieval gate's effect on unanswerable questions.
- A fixed-size chunking baseline.
- Whether differences on the 573-question sample are statistically significant.
