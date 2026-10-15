# What failed and how it was fixed

Every number here comes from a file in this repository: a report in `reports/`, the gate in
`configs/`, or an assertion in `tests/`. Retriever experiments use 40 dev contracts drawn with
seed 0 (573 answerable questions, `reports/experiments/`). The end-to-end answer run uses 10 dev
contracts drawn with seed 0 (410 questions, 126 answerable, `reports/answers/`). The only
test-split numbers are the final BM25 results. Retrieval metrics are over answerable questions at
k=10; a chunk counts as relevant if it overlaps an expert-annotated span.

## Shipped result

BM25 over SQLite FTS5, each question searched only inside its own contract, CUAD question text
verbatim (`reports/bm25-baseline-test.json`, 102 test contracts, 1,244 answerable questions):

| recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|
| 0.5514 | 0.6706 | 0.4780 | 0.5048 |

End to end with qwen2.5:7b through Ollama at temperature 0: BM25 top 10, the gate, generation,
quote grounding and the abstention rules (`reports/answers/dev-sample10-seed0-qwen2.5-7b.json`,
one run on dev). A shown answer is correct when one of its citations overlaps an expert span.

| answer recall | answer precision | false answer rate | citation precision |
|---|---|---|---|
| 0.4444 | 0.6747 | 0.0563 | 0.5470 |

With 126 answerable questions, answer recall lies between 0.36 and 0.53 (95% Wilson interval).

## Summary

| Failure | Fix | Status |
|---|---|---|
| Metadata questions share no words with their answers | Cross-encoder rerank tried | Open: helps metadata, hurts overall |
| Dense retrieval below BM25 | Hybrid fusion tried | Not shipped: no meaningful gain over BM25 |
| Cross-encoder rerank below BM25 overall | None | Not shipped |
| Clauses cut at chunk boundaries | Clause-aware chunking | Mitigated, no baseline to compare |
| Quotes the contract does not contain | Grounding every quote, strict abstention | Fixed: 7 of 126 model quotes rejected |
| Most questions are unanswerable | Retrieval gate plus abstention rules | Measured: 16 of 284 answered |
| Answerable clause never retrieved | None | Open: 38 of 70 misses |
| Model calls a retrieved clause absent | None | Open: 18 of 70 misses |

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

On the answer run, qwen2.5:7b claimed 126 quotes: 119 grounded, 10 of them by fuzzy matching, and
7 rejected. The rejections withheld 5 answerable and 2 unanswerable questions.

## 6. Most questions are unanswerable

Of 4,182 test questions, 2,938 have no answer in their contract
(`reports/bm25-baseline-test.json`). Two layers withhold answers:

- before the model: per-category BM25 score thresholds (`configs/bm25-gate.json`, 37 of 41
  categories), calibrated on dev to keep 0.95 of answerable questions. Raw BM25 scores are only
  comparable within one category, because each category asks one fixed question;
- after the model: withheld when the model reports the clause absent, cites nothing, or gives a
  quote that cannot be located.

On the answer run, of 284 unanswerable questions the gate withheld 103 before the model, at the
cost of 7 of 126 answerable ones, and spared 110 of 410 model calls. The model reported the clause
absent for 163 more, 2 were withheld for a quote that could not be located, and 16 were answered.
Of all withheld questions, 0.8196 have no answer in their contract.

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
- Observed: on a CPU, qwen2.5:7b takes 83.17 s per answer at the median and 135.16 s at p95. A
  reply slower than the fixed 300 s timeout was reported as "Ollama is not reachable", and the
  live tests hid the message. Timeouts are now named as such and set with `--timeout`; the answer
  run used 1800 s, and its one question that still timed out was retried on resume (both records
  are in the question log).

## 8. Where answerable questions are lost

Each of the 126 answerable questions of the answer run, by the stage that lost it. Retrieval is
replayed from the question log: BM25 is deterministic, and the replay reproduces every logged top
score (`tests/integration/test_answers_breakdown_cuad_release.py`).

| Outcome | Questions |
|---|---|
| Correct: a shown citation overlaps an expert span | 56 |
| No expert span in the 10 retrieved chunks | 38 |
| Retrieved, withheld by the gate | 5 |
| Retrieved, the model answers that the clause is absent | 18 |
| Retrieved, a quote cannot be located | 4 |
| Retrieved, the answer cites other text | 5 |

Retrieval loses more answerable questions (38) than all later stages together (32): whatever
follows retrieval, at most 88 of 126 can be answered. The metadata categories of section 1 fail
end to end too: Parties 0 of 10 correct, Document Name 0 of 10, Agreement Date 1 of 10. None of
the 5 Parties answers shown cites an annotated party name.

## Not measured

- Answers on the test split, with another model, or across repeated runs: the answer run is one
  run of qwen2.5:7b on dev.
- A fixed-size chunking baseline.
- Whether differences on the 573-question sample are statistically significant.
