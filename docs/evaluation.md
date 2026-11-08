# DocMind AI: Evaluation

> **Status:** evaluated with real OpenAI models on 2026-11-08. Result: **12/12** on this small set, and every answer was checked by hand against its cited chunk (see §4). The raw generated report is [evaluation-results.md](evaluation-results.md).

## 1. Evaluation questions with expected answers

The corpus is the fictional sample set in `data/sample_docs/`. Expected answers come directly from the source text.

| ID | Question | Expected answer (key facts) | Relevant source | Keywords checked |
|---|---|---|---|---|
| Q1 | How many vacation days do full-time employees get, and can unused days be carried over? | 28 days per year; up to 5 days carry over, expiring 31 March | `employee_handbook.md`: Paid Time Off | `28`, `5` |
| Q2 | How quickly must a phishing email be reported, and to whom? | Within one hour, to security@acme-analytics.example (forward as attachment) | `it_security_policy.md`: Incident Reporting | `one hour`, `security@acme-analytics.example` |
| Q3 | How much does the Professional plan cost? | 79 EUR per user per month (15% off with annual billing) | `product_faq.txt`: pricing plans | `79` |
| Q4 | Where is customer data stored? | Data centres in Frankfurt, Germany; never leaves the EU | `product_faq.txt`: data storage | `Frankfurt` |
| Q5 | How many days per week can employees work remotely? | Up to three days per week | `employee_handbook.md`: Remote Work Policy | `three` |
| Q6 | What is the minimum password length? | 14 characters | `it_security_policy.md`: Passwords and Authentication | `14` |
| Q7 | What happens to workspace data after a subscription is cancelled? | Kept 30 days for export/restore, then deleted; backups purged within a further 60 days | `product_faq.txt`: data retention | `30 days` |
| Q8 | Is there a native Salesforce connector? | No, planned but not yet available | `product_faq.txt`: data sources | `not yet` |

Q8 is deliberately a **"negative but answerable"** question: the correct answer is "no", and it *is* in the documents. A system that refuses too eagerly would fail it.

## 2. Questions without supporting context

| ID | Question | Why it is unanswerable | Expected behaviour |
|---|---|---|---|
| N1 | What is the name of the CEO of Acme Analytics? | No person is named anywhere | Insufficient-context response |
| N2 | What is the current stock price of Acme Analytics? | Not in corpus; also real-time data | Insufficient-context response |
| N3 | What will the weather be like in Berlin tomorrow? | Completely off-topic | Insufficient-context response, ideally blocked by the distance gate before the LLM is called |
| N4 | How many vacation days do interns get? | **Near-miss:** the handbook covers vacation for permanent full-time/part-time employees, and the security policy mentions interns, but nothing states intern vacation | Insufficient-context response. Answering "28" would be a hallucination by over-generalisation |

N4 is the hardest case: retrieval *will* find relevant-looking vacation chunks, so only the LLM-level guardrail can catch it.

## 3. How retrieval and response quality are assessed

`evaluate.py` runs every question through the real pipeline and reports three transparent metrics:

| Metric | Definition | Applies to |
|---|---|---|
| **Retrieval hit** | The expected source file is among the chunks that passed the distance threshold and were sent to the LLM | Q1–Q8 |
| **Answer keyword match** | The answer is marked sufficient **and** contains every expected keyword (case-insensitive) | Q1–Q8 |
| **Correct refusal** | The pipeline returns the insufficient-context response | N1–N4 |

A question **passes** if (answerable) retrieval hit and keyword match are both true, or (unanswerable) the refusal is correct. The report also records the full answer text, the cited chunks and the top retrieved chunk with its similarity, so each verdict can be checked by hand.

**Manual review checklist** (for each answer in the generated report):
1. *Faithfulness*: is every claim supported by the cited chunk?
2. *Citation accuracy*: do the `[S#]` labels point at the chunk that contains the fact?
3. *Completeness*: for two-part questions (Q1, Q2, Q7), are both parts answered?
4. *Calibration*: inspect similarities for N1–N4 vs. Q1–Q8 to choose `DOCMIND_MAX_DISTANCE`.

## 4. Actual evaluation results

### Automated tests: run ✅

```text
$ pytest   (Python 3.12.10, Windows 11)
22 passed in 1.17s
```

These tests check the **mechanics** with a fake bag-of-words embedder and a fake LLM against a real ChromaDB instance: chunking invariants, loading rules, batching/ordering of embeddings, metadata storage, re-ingestion idempotence, citation mapping, both insufficient-context guardrails (including that the LLM is **not** called when nothing is relevant), and error cases. They do **not** measure answer quality.

### Live model evaluation: run ✅

Run: 2026-11-08 10:22 UTC · `text-embedding-3-small` + `gpt-4o-mini` · top_k=4, max_distance=0.70, chunk_size=800, overlap=150. Re-run with `python ingest.py && python evaluate.py`.

| Metric | Result |
|---|---|
| Retrieval hit rate (Q1–Q8) | **8/8** |
| Answer keyword match (Q1–Q8) | **8/8** |
| Correct refusals (N1–N4) | **4/4** |
| Overall | **12/12** |

**Manual verification.** Each answer was compared with the text of the chunks it cites:

| ID | Answer correct? | Citation points at the supporting chunk? | Notes |
|---|---|---|---|
| Q1 | ✅ 28 days; 5 carry over; expire 31 March | ✅ handbook#2 holds all three facts | Both parts of the question answered |
| Q2 | ✅ one hour, security@acme-analytics.example | ✅ security#3 | Omits "forward as attachment" (not asked) |
| Q3 | ✅ 79 EUR / user / month | ✅ faq#0 | The **top-ranked** chunk (faq#1) holds only the end of the pricing section, without "79". The model correctly cited the 2nd-ranked chunk. This shows why top_k > 1 matters |
| Q4 | ✅ Frankfurt, EU only, TLS 1.2+, AES-256 | ✅ faq#2 and faq#3 | The fact sits on a chunk boundary; overlap put it in both chunks. Top-ranked chunk was security#2 (EU data classification), retrieved but correctly not cited |
| Q5 | ✅ three days | ✅ handbook#1 | Highest similarity in the set (0.66) |
| Q6 | ✅ 14 characters | ✅ security#0 | |
| Q7 | ✅ 30 days, then deleted; backups +60 days | ✅ faq#3 | Both parts answered |
| Q8 | ✅ "No, planned but not yet available" | ✅ faq#2 and faq#1 | Both cited chunks contain the sentence (overlap). Not wrongly refused |
| N1–N4 | ✅ refused | n/a | See guardrail analysis below |

No hallucinated facts or wrong citations were found.

**Which guardrail refused each unanswerable question:**

| ID | Top similarity | Passed distance gate? | Refused by |
|---|---|---|---|
| N1 CEO name | 0.52 | yes | LLM (`INSUFFICIENT_CONTEXT`) |
| N2 stock price | 0.55 | yes | LLM |
| N3 weather | 0.20 | **no** | Distance gate (LLM not called) |
| N4 intern vacation | 0.43 | yes | LLM, despite vacation chunks being retrieved |

**Threshold finding.** Top similarities for answerable questions ranged from **0.47 to 0.66**. The on-topic unanswerable questions N1/N2 scored **0.52–0.55**, *higher* than Q2, Q3 and Q6 (0.47–0.48). Similarity measures *topic*, not *whether the answer is present*. So:
- the distance gate is useful only for clearly off-topic questions (N3 at 0.20), where it saves an LLM call;
- tightening it enough to catch N1/N2 would also reject correct answers such as Q2 and Q3;
- the LLM-level check is the guardrail doing the real work. `max_distance=0.70` (similarity ≥ 0.30) was kept.

The threshold filters **each chunk separately**, not just the best match. In a live `--show-context` run for Q2, the 4th chunk at distance 0.706 was dropped from the prompt while the three below 0.70 were sent. The CLI now labels each chunk `[sent to LLM]` or `[FILTERED]`.

## 5. Limitations of the evaluation

- **Tiny, hand-written set** (12 questions, 3 short documents). Useful as a smoke test; not statistically meaningful.
- **Written by the system's author**, who knows the corpus, so the questions may be easier than real user questions (fewer paraphrases, typos or ambiguity).
- **Keyword matching is crude.** It can pass a wrong answer that happens to contain "28", or fail a correct one that writes "3" instead of "three". Manual review of the generated report is still required.
- **File-level retrieval hits** don't check that the *specific* chunk with the fact was retrieved, only that some chunk of the right file was.
- **No faithfulness scoring.** Citations are mapped structurally; whether each claim is supported is judged by hand. Proposed: LLM-as-judge / RAGAS faithfulness and answer relevance.
- **Non-determinism.** Even at temperature 0, model outputs can vary between runs and model versions; results should be recorded with date and model name.
- **The threshold was reviewed on the same set it is evaluated on**, which can overstate performance; a held-out set would be needed for a fair number.
- **12/12 is not an accuracy guarantee.** The corpus is short, clean and unambiguous. Real documents (contradictions, tables, long PDFs, outdated versions) would be much harder.
- **Single run.** Each question was asked once; repeated runs would show how stable the answers are.
