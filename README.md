# Self-Corrective RAG with Adaptive Evidence Verification
### Hallucination-Resistant Answer Generation

A Retrieval-Augmented Generation system that doesn't just retrieve-and-generate — it
**grades its own evidence, rewrites bad queries, falls back to web search, and checks
its own answers for hallucination and relevance before returning them to the user.**

This is a from-scratch implementation of the "Corrective RAG" / "Self-RAG" family of
ideas, built with **LangGraph** as a cyclic state machine (not a simple linear chain).

---

## 1. Why this is different from plain RAG

Plain RAG:
```
question -> retrieve top-k chunks -> stuff into prompt -> generate -> return
```
Problem: if retrieval returns irrelevant/weak chunks, the LLM still generates a
confident-sounding answer from bad evidence -> hallucination.

This project instead runs a **graph with feedback loops**:

```
                     ┌────────────────────────────────────────┐
                     ▼                                        │
question ──► retrieve ──► grade_documents ──► [relevant?] ──► generate
                              │  no / weak                        │
                              ▼                                   ▼
                       transform_query                    grade_hallucination
                              │                                   │
                              ▼                          [grounded in docs?]
                        (loop back to                      no │        │ yes
                         retrieve, up to                      ▼        ▼
                         MAX_RETRIES)                    regenerate  grade_answer
                              │                                       │
                              ▼                              [answers question?]
                        web_search_fallback                   no │        │ yes
                        (if still weak after                     ▼        ▼
                         retries)                          transform_query  RETURN
                                                             (loop, bounded)
```

Every loop is **bounded** (`MAX_RETRIES` in `config.py`) so the graph always
terminates — it either returns a verified answer or an honest "I don't have
enough evidence to answer that confidently" response.

## 2. Pipeline stages (maps to `src/`)

| Stage | File | What it does |
|---|---|---|
| Ingestion | `src/ingestion.py` | Loads `.txt`/`.pdf`/`.md` docs, chunks them (recursive char splitter) |
| Vector store | `src/vectorstore.py` | Embeds chunks (OpenAI or local HF embeddings) into Chroma |
| Retrieval | `src/graph.py::retrieve` | Top-k similarity search |
| **Evidence grading** | `src/grading.py::grade_documents` | LLM grades each chunk as relevant/irrelevant to the question (structured output) |
| **Query rewriting** | `src/query_rewriter.py` | If evidence is weak, rewrites the question for better retrieval (not a blind retry) |
| **Web fallback** | `src/web_search.py` | If local corpus still doesn't cover the question after retries, optionally pulls in web results (Tavily) |
| Generation | `src/generation.py::generate_answer` | Generates an answer strictly from the graded, relevant evidence |
| **Hallucination check** | `src/verification.py::grade_hallucination` | LLM checks: is every claim in the answer actually supported by the retrieved documents? |
| **Answer-quality check** | `src/verification.py::grade_answer` | LLM checks: does the answer actually address the question asked? |
| Orchestration | `src/graph.py` | Wires all of the above into a LangGraph `StateGraph` with conditional edges |

## 3. Setup

```bash
cd self-corrective-rag
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill in your keys
```

`.env` needs at minimum:
```
OPENAI_API_KEY=sk-...
```
Optional (enables the web-search fallback node):
```
TAVILY_API_KEY=tvly-...
```
No OpenAI key? Set `LLM_PROVIDER=local` and `EMBEDDING_PROVIDER=local` in `.env` to
run entirely on local HuggingFace models via `src/config.py` (slower, no API cost).

## 4. Usage

**1. Ingest your documents** (drop files into `data/sample_docs/` first, or point
   `--source` at any folder):
```bash
python app.py ingest --source data/sample_docs
```

**2. Ask a question:**
```bash
python app.py ask "What triggers the query rewrite step in this pipeline?"
```

**3. Run interactively:**
```bash
python app.py chat
```

**4. Inspect a run's full trace** (which nodes fired, grades given, retry count):
```bash
python app.py ask "..." --trace
```

## 5. Project layout

```
self-corrective-rag/
├── app.py                    # CLI entrypoint (ingest / ask / chat)
├── requirements.txt
├── .env.example
├── data/
│   └── sample_docs/          # put your source documents here
├── src/
│   ├── config.py             # env vars, model names, thresholds, MAX_RETRIES
│   ├── state.py              # the GraphState TypedDict passed between nodes
│   ├── ingestion.py          # load + chunk documents
│   ├── vectorstore.py        # Chroma persistence layer
│   ├── grading.py            # document relevance grader (structured LLM output)
│   ├── query_rewriter.py     # question rewriting node
│   ├── generation.py         # answer generation node
│   ├── verification.py       # hallucination + answer-quality graders
│   ├── web_search.py         # Tavily fallback retrieval
│   └── graph.py              # LangGraph StateGraph wiring + conditional routing
└── tests/
    └── test_pipeline.py      # unit tests for graders/routing with mocked LLM
```

## 6. Key design decisions worth mentioning in your report/demo

- **Structured grading, not vibes**: every grader (`grading.py`, `verification.py`)
  forces the LLM to return a Pydantic model (`binary_score: "yes"/"no"` + reasoning),
  not free text — this makes routing deterministic and testable.
- **Bounded loops**: `state["retries"]` is incremented on every corrective loop and
  checked against `MAX_RETRIES` in `config.py`, so the graph can never spin forever.
- **Separation of "is this evidence relevant" vs "is this answer grounded"**: these
  are two different failure modes (bad retrieval vs. good retrieval + LLM still
  drifts) and are graded by two separate nodes.
- **Graceful degradation**: if hallucination/answer checks keep failing after
  `MAX_RETRIES`, the system returns a hedged answer that says explicitly which
  claims are supported and which are not, rather than silently returning a
  possibly-wrong confident answer.

## 7. Extending this

- Swap Chroma for FAISS/Pinecone/Weaviate in `src/vectorstore.py` — the rest of the
  graph doesn't care about the backing store.
- Add a `src/reranker.py` node (e.g. Cohere rerank or a cross-encoder) between
  `retrieve` and `grade_documents` for a stronger first-pass filter.
- Swap the grader LLM for a cheaper/smaller model than the generator LLM — grading
  is a much easier task than generation (see `GRADER_MODEL` vs `GENERATOR_MODEL`
  in `config.py`).
