# DocMind AI: Project Plan

## 1. Project scope

**Goal:** show a working, tested and well-documented RAG pipeline in Python as a portfolio proof of concept.

**In scope (delivered)**
- Load `.txt` / `.md` from a local folder
- Overlapping chunking; OpenAI embeddings; ChromaDB persistence with source metadata
- CLI to ingest and to ask questions; answers with traceable citations
- Honest insufficient-context responses
- Automated tests with mocked AI calls; evaluation script and question set
- Technical documentation (README, architecture, plan, evaluation)

**Out of scope (documented as future work)**
- REST API, web UI, additional LLM providers, MCP server
- Cloud deployment, Docker, CI/CD
- Authentication, document permissions
- Large-scale ingestion, advanced evaluation and monitoring

## 2. Work Breakdown Structure (WBS)

```text
1 DocMind AI PoC
├── 1.1 Setup
│   ├── 1.1.1 Repo structure, .gitignore, requirements, pyproject (pytest config)
│   └── 1.1.2 Configuration via env vars / .env (+ .env.example)
├── 1.2 Ingestion
│   ├── 1.2.1 Document loader (.txt/.md, recursive, deterministic)
│   ├── 1.2.2 Chunker (overlap, boundary detection, offsets)
│   ├── 1.2.3 OpenAI embedding wrapper (batching, ordering)
│   └── 1.2.4 ChromaDB store (reset, add, search, cosine)
├── 1.3 Question answering
│   ├── 1.3.1 Retrieval + relevance threshold
│   ├── 1.3.2 Prompt design (grounding, citations, refusal token)
│   ├── 1.3.3 Chat wrapper
│   └── 1.3.4 Citation mapping + insufficient-context handling
├── 1.4 Interfaces
│   ├── 1.4.1 ingest.py
│   ├── 1.4.2 ask.py (--top-k, --show-context)
│   └── 1.4.3 evaluate.py (report generation)
├── 1.5 Quality
│   ├── 1.5.1 Unit tests (chunker, loader, model wrappers)
│   ├── 1.5.2 Pipeline tests (real Chroma, fake models)
│   └── 1.5.3 Evaluation question set
└── 1.6 Documentation
    ├── 1.6.1 README
    ├── 1.6.2 Architecture
    ├── 1.6.3 Project plan
    └── 1.6.4 Evaluation
```

## 3. Milestones

| # | Milestone | Exit criterion | Status |
|---|---|---|---|
| M1 | Skeleton & config | Package imports; settings load from `.env` | ✅ Done |
| M2 | Ingestion pipeline | Sample docs → 15 chunks stored in Chroma (verified with fake embedder) | ✅ Done |
| M3 | Grounded Q&A | `ask.py` returns cited answer or insufficient-context reply | ✅ Done (verified live) |
| M4 | Tests | All tests pass offline | ✅ Done (22 tests) |
| M5 | Documentation | README + 3 docs complete | ✅ Done |
| M6 | Live evaluation | `evaluate.py` run with real models; results recorded; threshold reviewed | ✅ Done: 12/12, manually verified; threshold kept at 0.70 |

## 4. Technical decisions

| Decision | Rationale | Revisit when |
|---|---|---|
| No orchestration framework (LangChain etc.) | Transparency; less code than the abstraction; easier to test and explain | Need agents, many integrations, or complex chains |
| ChromaDB embedded | Zero infrastructure | Multi-user access, >1M chunks, or managed hosting needed |
| `text-embedding-3-small` | Low cost, good quality | Retrieval quality is the bottleneck in evaluation |
| `gpt-4o-mini`, temperature 0 | Cheap, consistent, follows formatting | Complex multi-source reasoning needed |
| Character chunking 800/150 | Simple, deterministic, fits short policy sections | Long technical docs or structured Markdown |
| Full re-index on ingest | Simple and idempotent | Corpus large enough that re-embedding is slow or costly |
| Two-layer refusal (distance gate + LLM token) | Saves cost on clearly irrelevant questions; LLM handles subtle cases | Calibration shows the gate causes false refusals |
| Protocol-based DI | Offline, deterministic tests | — |

## 5. Risks and dependencies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Hallucinated or unsupported answers | Medium | High | Grounding prompt, refusal token, citation display; future claim verification |
| Poorly chosen relevance threshold (false refusals / weak context passed) | Medium | Medium | Configurable; `--show-context` shows distances; calibrate in M6 |
| OpenAI outage / rate limits / price or model changes | Low–Med | Medium | SDK retries; models configurable by env var; provider protocols allow switching |
| Sensitive data sent to a third-party API | Medium | High | Fictional sample data; documented data-egress warning; future Azure OpenAI / local models |
| Prompt injection via documents | Low (PoC) | Medium | Untrusted-source instruction; model has no tools |
| ChromaDB API changes between versions | Medium | Low | Version range pinned in `requirements.txt`; thin wrapper isolates usage |

**Dependencies:** OpenAI API account and key; Python 3.12+; `openai`, `chromadb`, `python-dotenv`, `pytest`.

## 6. Success criteria

| Criterion | Target | Current |
|---|---|---|
| End-to-end CLI works (`ingest.py`, `ask.py`) | Yes | ✅ Verified live with OpenAI |
| Answers include traceable sources | File + chunk + char range | ✅ Implemented, tested, and citations verified by hand |
| Unanswerable questions refused honestly | ≥ 3/4 on eval set | ✅ 4/4 |
| Retrieval hit rate on answerable eval questions | ≥ 7/8 | ✅ 8/8 |
| Automated tests pass offline | 100% | ✅ 22/22 |
| Documentation separates implemented from proposed | Yes | ✅ |

## 7. Estimated effort

Estimates are for one developer who is comfortable with APIs and new to Python.

| Work package | Estimate |
|---|---|
| 1.1 Setup & config | 0.5 day |
| 1.2 Ingestion | 1 day |
| 1.3 Question answering & prompts | 1 day |
| 1.4 CLIs + evaluation script | 0.5 day |
| 1.5 Tests | 1 day |
| 1.6 Documentation | 1 day |
| M6 Live evaluation & threshold calibration | 0.5 day |
| **Total** | **≈ 5.5 days** |

## 8. Future roadmap (not implemented)

| Phase | Item | Notes | Rough effort |
|---|---|---|---|
| 1 | Threshold calibration & hybrid search | Calibrate `max_distance` on real scores; add BM25 + vector fusion; optional re-ranker | 2–3 days |
| 1 | Advanced evaluation & monitoring | Larger labelled set, RAGAS / LLM-as-judge (faithfulness, answer relevance), latency/cost logging, tracing | 3–5 days |
| 2 | FastAPI REST backend | `/ingest`, `/ask`, `/health`; Pydantic models; async OpenAI client | 2 days |
| 2 | React frontend | Chat UI, citations as clickable highlighted spans using stored offsets | 3–4 days |
| 2 | Gemini provider integration | New `Embedder`/`ChatModel` implementations; provider selected by config. Note that embeddings from different providers are not interchangeable (re-index needed) | 1 day |
| 2 | MCP server | `search_documents` / `ask_documents` tools, document resources (see architecture §8) | 1–2 days |
| 3 | Larger-scale ingestion | PDF/DOCX/HTML parsers, incremental updates via content hashes, background jobs, managed vector DB (pgvector / Azure AI Search) | 1–2 weeks |
| 3 | Authentication & document permissions | SSO (e.g. Entra ID), ACL metadata per chunk, retrieval filtered by user groups | 1–2 weeks |
| 3 | Azure deployment | Azure OpenAI (data residency), Container Apps / App Service, Key Vault for secrets, AI Search as vector store | 1 week |
