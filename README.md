# Teaching RAG with LangGraph — Helios Bikes Example

A hands-on example for teaching Retrieval-Augmented Generation (RAG) built as a
**LangGraph** graph, using **LangChain** components, **Voyage AI** embeddings,
and **Claude** for generation.

<img width="767" height="812" alt="image" src="https://github.com/user-attachments/assets/241ae288-6f73-44fe-9c99-ab852093bd59" />

## What's in this folder

```
langchain_tutorial/
├── app.py                # Streamlit chat UI + corrective RAG graph
├── docs/                 # the knowledge base — 6 short documents
│   ├── 01_company.txt
│   ├── 02_products.txt
│   ├── 03_warranty.txt
│   ├── 04_battery_care.txt
│   ├── 05_shipping.txt
│   └── 06_support.txt
├── requirements.txt
├── handout.html          # one-page slide-ready reference
└── README.md             # this file
```

The documents describe a fictional electric-bike company, *Helios Bikes*. They
are deliberately written so that facts are **spread across documents**, which is
what makes retrieval interesting to teach.

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."   # from console.anthropic.com
export VOYAGE_API_KEY="pa-..."          # from dash.voyageai.com

streamlit run app.py
```

Two keys are needed because **Claude has no embeddings API**: Voyage does the
embeddings, Claude does the generation. Two providers, two bills.

## The two ideas to teach

### 1. RAG = Retrieve, then Generate

Before answering, the system *retrieves* relevant text from the documents and
hands it to the model as context. The model answers only from that context, so
it stays grounded instead of guessing. The retrieval works on **embeddings**:
every chunk of text is turned into a vector, and the question is matched to the
closest chunks by meaning, not keywords.

The indexing half (run once):

```
documents → split into chunks → embed each chunk → store the vectors
```

The query half (every question):

```
embed the question → find nearest chunks → put them in Claude's prompt → answer
```

### 2. Why LangGraph instead of a plain chain

A plain chain is a straight line: retrieve → answer. It can't react if retrieval
fails. This example is a **graph** that can *loop*: if the retrieved documents
don't actually contain the answer, it rewrites the question and tries again.

```
        START
          │
          ▼
   ┌──────────────┐
   │ contextualize│  (rewrite follow-ups into standalone queries)
   └──────────────┘
          │
          ▼
      ┌────────┐
      │retrieve│◄─────────────┐
      └────────┘              │
          │                   │
          ▼                   │
      ┌────────┐              │
      │ grade  │ relevant?    │
      └────────┘              │
          │                   │
   ┌──────┼─────────┐         │
   ▼      ▼         ▼         │
generate rewrite  give_up     │
   │      │         │         │
   │      └─────────┼─────────┘   (loop back)
   ▼                ▼
  END              END
```

The arrow from **rewrite back to retrieve** is the whole point. That cycle is
something a linear LangChain pipeline cannot express — and it's why LangGraph
exists.

## LangGraph concepts in the code

| Concept | Where in the code | What it is |
|---|---|---|
| **State** | `class ChatState` | a shared dict every node reads and updates |
| **Nodes** | `contextualize`, `retrieve`, `grade`, `rewrite`, `generate`, `give_up` | functions: state in, update out |
| **Edges** | `add_edge(...)` | fixed "always go here next" transitions |
| **Conditional edges** | `add_conditional_edges("grade", decide_next, ...)` | branch based on the state |
| **Cycle** | `add_edge("rewrite", "retrieve")` | loop back to try again |
| **Memory** | `InMemorySaver` + `thread_id` | conversation persists across turns |

## Explanations

Walk through the **Agent trace** panel step by step. Each row below matches one
trace card in the UI and maps to concrete functions in `app.py`.

### Before the graph runs: indexing (once at startup)

When Streamlit loads, `@st.cache_resource` calls `build_retriever()`. This is
plain LangChain — no LangGraph yet:

| Step | What happens | LangChain / LangGraph |
|---|---|---|
| Load files | Read each `.txt` in `docs/` | `Document(page_content=..., metadata={"source": ...})` |
| Split | Cut docs into overlapping chunks | `RecursiveCharacterTextSplitter.split_documents()` |
| Embed | Turn each chunk into a vector | `VoyageAIEmbeddings(model="voyage-3-large")` |
| Index | Store vectors in memory | `InMemoryVectorStore.from_documents()` |
| Expose search | Return a search interface | `.as_retriever(search_kwargs={"k": TOP_K})` |

**Key idea:** embeddings measure *similarity*, not *correctness*. A chunk can
rank highly and still not contain the answer — that is why we need a **grade**
step later.

### Memory across chat turns (between questions)

Each new message re-enters the graph, but the bot remembers prior turns:

| Piece | Role | Where |
|---|---|---|
| `ChatState.messages` | Holds the full conversation | `Annotated[list, add_messages]` — LangGraph **reducer** that *appends* new messages instead of overwriting |
| `InMemorySaver()` | Saves graph state per session | Passed to `builder.compile(checkpointer=...)` |
| `thread_id` | Identifies one conversation | `config={"configurable": {"thread_id": ...}}` on every `app.stream()` call |

**Key idea:** we pass only the *new* `HumanMessage` each turn; the checkpointer
reloads prior state. The **contextualize** node resets per-turn counters
(`attempts`, `relevant`) so the correction loop starts fresh every question.

### The graph, node by node

Every node is a plain Python function registered with
`builder.add_node(...)`. Each returns a partial dict that LangGraph merges into
`ChatState`.

#### 1. Contextualize

**What to say:** RAG searches by embedding a *question*. Follow-ups like
*"What about the Hauler?"* embed poorly on their own. This step rewrites them
into a standalone query using chat history.

| | |
|---|---|
| **LangGraph** | Node `contextualize`; fixed edge `START → contextualize → retrieve` |
| **LangChain** | `contextualize_chain` — a `ChatPromptTemplate` (with `MessagesPlaceholder("history")`) piped to `ChatAnthropic` via `\| llm` (LCEL) |
| **First message** | Skips the LLM call — the question is already standalone |
| **Follow-ups** | `contextualize_chain.invoke({"history": ..., "followup": ...})` |

#### 2. Retrieve

**What to say:** This is classic RAG retrieval — embed the query, find the
nearest chunk vectors. It does **not** call Claude. Voyage embeddings do the work.

| | |
|---|---|
| **LangGraph** | Node `retrieve`; increments `attempts` each time it runs |
| **LangChain** | `retriever.invoke(state["question"])` — under the hood: embed query with `VoyageAIEmbeddings`, cosine-search `InMemoryVectorStore`, return top `TOP_K` chunks |
| **Output** | Chunk text → `state["documents"]`; source filenames → trace previews |

**Key idea:** retrieval answers *"what looks similar?"* not *"does this answer
the question?"*

#### 3. Grade

**What to say:** Retrieved chunks might be irrelevant (e.g. bike docs for a
scooter question). Before answering, we ask Claude to judge yes/no.

| | |
|---|---|
| **LangGraph** | Node `grade`; then `add_conditional_edges("grade", decide_next, ...)` picks the next node |
| **LangChain** | `grader_chain` — `ChatPromptTemplate` + `ChatAnthropic`; sends `QUESTION` + joined `CONTEXT` chunks; expects one word: `yes` or `no` |
| **Routing logic** | Plain Python in `decide_next()`: `verdict.startswith("yes")` → `relevant = True` |

**Key idea:** the grader is a *second* LLM call with a narrow job — quality
control between retrieval and generation.

#### 4. Rewrite (part of the correction loop)

**What to say:** If the grader says *no* and we still have retries left, don't
give up yet — ask Claude to rephrase the search query with different keywords,
then search again.

| | |
|---|---|
| **LangGraph** | Node `rewrite`; fixed edge `rewrite → retrieve` (**the cycle**) |
| **LangChain** | `rewrite_chain` — `ChatPromptTemplate` + `ChatAnthropic`; input: original question + query that failed |
| **Output** | Updates `state["question"]` to the new search string |

#### 5. Generate

**What to say:** Grader said *yes* — safe to answer. Claude writes the reply
grounded in retrieved chunks **and** the conversation so far.

| | |
|---|---|
| **LangGraph** | Node `generate`; edge `generate → END` |
| **LangChain** | `answer_chain` — `ChatPromptTemplate` with `MessagesPlaceholder("history")` + context block; piped to `ChatAnthropic` |
| **Output** | `{"messages": [AIMessage(content=answer)]}` — appended via `add_messages` reducer |

#### 6. Give up

**What to say:** After `MAX_ATTEMPTS` retrievals, stop looping. Return a fixed
fallback — never invent an answer.

| | |
|---|---|
| **LangGraph** | Node `give_up`; edge `give_up → END`; reached when `decide_next()` sees `attempts >= MAX_ATTEMPTS` |
| **LangChain** | None — no LLM call, just a hard-coded `AIMessage` |
| **Why it matters** | Shows graceful failure vs hallucination |

### The correction loop — explain this carefully

This is the main reason to use LangGraph instead of a linear chain.

```
retrieve → grade
              │
     relevant? ├── yes ──→ generate → END
              │
              └── no ──→ attempts < MAX_ATTEMPTS?
                        ├── yes ──→ rewrite ──→ retrieve  (loop)
                        └── no  ──→ give_up → END
```

| LangGraph mechanism | What it does |
|---|---|
| `add_conditional_edges("grade", decide_next, {...})` | After grading, call `decide_next(state)` to pick `"generate"`, `"rewrite"`, or `"give_up"` |
| `add_edge("rewrite", "retrieve")` | The **cycle** — rewrite feeds back into retrieve |
| `state["attempts"]` | Incremented inside `retrieve`; compared to `MAX_ATTEMPTS` (default `2`) in `decide_next` |
| `app.stream(..., stream_mode="updates")` | Streamlit emits one trace card per node as the loop runs — students can watch attempt 1 and attempt 2 live |

**Walk through a bad-case question** (e.g. *"Do you sell electric scooters?"*):

1. **Retrieve attempt 1** — embedding search returns bike chunks (similar topic, wrong product).
2. **Grade** — `grader_chain` says `no` (chunks don't mention scooters).
3. **Rewrite** — `rewrite_chain` produces a new search query.
4. **Retrieve attempt 2** — still no scooter docs in the knowledge base.
5. **Grade** — `no` again; `attempts` is now `2`.
6. **Give up** — `decide_next` routes to `give_up` instead of looping forever.

**Walk through a good-case question** (e.g. *"How much does the Trailblazer cost?"*):

1. **Contextualize** — already standalone, no change.
2. **Retrieve** — `02_products.txt` chunks rank high.
3. **Grade** — `yes`.
4. **Generate** — `answer_chain` produces the price. No rewrite, no loop.

### LangChain vs LangGraph — one sentence each

- **LangChain** supplies the building blocks: prompts (`ChatPromptTemplate`),
  models (`ChatAnthropic`), embeddings (`VoyageAIEmbeddings`), vector store
  (`InMemoryVectorStore`), text splitting, and the LCEL pipe operator (`|`).
- **LangGraph** supplies the *control flow*: `StateGraph`, nodes, edges,
  conditional routing, cycles, checkpointed memory, and streaming updates.

The chains do the *thinking*; the graph decides *what to run next*.

## Demo questions

Run `streamlit run app.py` and use the **Agent trace** panel on the right. Each
question below is chosen to show a different behavior. Ask them in order the
first time you teach; later, mix and match.

### Good cases — the answer is in the knowledge base

These should reach **generate** (possibly after **contextualize** on follow-ups).
Point students at which source file(s) appear in the retrieve step.

| # | Question | What to teach | Expected sources |
|---|---|---|---|
| 1 | **How much does the Trailblazer cost?** | Simplest path: one fact, one doc, no loop. | `02_products.txt` |
| 2 | **What's the warranty on the battery in the Hauler?** | Multi-hop: catalog says all bikes use the PowerCell; warranty doc gives the 2-year battery term. | `02_products.txt`, `03_warranty.txt` |
| 3 | **How should I look after my bike's battery during the winter?** | Semantic search, not keywords — "look after" / "winter" vs "winter storage" in the source. | `04_battery_care.txt` |
| 4 | **Do you ship to Texas?** | Answer is implied (all 50 states), not a literal "Texas" mention — tests grounding vs guessing. | `05_shipping.txt` |
| 5 | **What are your store hours in Portland?** | Precise fact lookup from support doc. | `06_support.txt` |
| 6 | **Can I return a bike if I've ridden 30 miles?** | Reasoning over a policy threshold (returns allowed under 50 miles). | `03_warranty.txt` |

**Follow-up conversation** (ask as a second and third message in the *same*
session — do not click "New conversation"):

| Turn | Question | What to teach |
|---|---|---|
| 1 | **How much does the Trailblazer cost?** | Baseline answer. |
| 2 | **What about the Hauler?** | **Contextualize** resolves "cost" from history; watch the trace rewrite the follow-up into a standalone query. |
| 3 | **And how heavy is it?** | **Contextualize** resolves "it" → Hauler; retrieval switches to weight in `02_products.txt`. |
| 4 | **Is its battery covered by warranty?** | Memory + multi-doc again (PowerCell platform + 2-year battery warranty). |

### Bad cases — the answer is *not* in the knowledge base

These should **not** invent facts. After **grade** finds nothing useful, the
graph **rewrites** the query, retrieves again, then **give_up** (with
`MAX_ATTEMPTS = 2`). The bot should say it couldn't find the information.

| # | Question | What to teach | Why it's unanswerable |
|---|---|---|---|
| 1 | **Do you sell electric scooters?** | Classic out-of-scope product — Helios sells three *bikes* only. | No scooters in any doc |
| 2 | **What's your stock price?** | Company is privately held; model should refuse, not fabricate a ticker. | `01_company.txt` says not publicly traded |
| 3 | **Do you ship to Canada?** | Explicit negative in shipping doc (U.S. only). | `05_shipping.txt` — no international shipping |
| 4 | **Can I finance a bike with monthly payments?** | Common real-world question with zero coverage — tests hallucination resistance. | No financing doc exists |
| 5 | **What color options does the Dart come in?** | Catalog lists specs and price, not colors. | Not in knowledge base |

**What students should see in the trace for bad cases:**

```
contextualize → retrieve → grade (not relevant)
  → rewrite → retrieve → grade (still not relevant) → give_up
```

Contrast a **good** case (straight or short path):

```
contextualize → retrieve → grade (relevant) → generate
```

### Quick reference — which path for which question?

| Path | Example questions |
|---|---|
| **Straight hit** | Trailblazer price, Portland store hours |
| **Multi-document** | Hauler battery warranty, follow-up "Is its battery covered?" |
| **Semantic match** | Winter battery care, "look after" wording |
| **Implied fact** | Ship to Texas (all 50 states) |
| **Rewrite → give up** | Scooters, financing, Canada shipping, stock price |

## Suggested classroom exercises

- **Good vs bad side by side:** ask question 1 from each table above (Trailblazer
  price, then electric scooters). Compare the trace — one ends at **generate**,
  the other at **give_up**.
- **Break retrieval on purpose:** in `app.py`, set `TOP_K = 1` and ask the Hauler
  battery warranty question. Watch it miss one of two needed docs and trigger the
  rewrite loop.
- **Turn off the loop:** set `MAX_ATTEMPTS = 1` and ask the scooter question —
  it gives up after one attempt instead of rewriting. This isolates what the cycle
  buys you.
- **Follow-up without memory:** click **New conversation**, then ask *"What about
  the Hauler?"* with no prior context. Compare to the three-turn follow-up demo.
- **Add a document:** drop a new `.txt` into `docs/` (e.g. a financing/payment
  plan), restart Streamlit, and ask about it — the bad-case financing question
  should suddenly become answerable.
- **Inspect chunking:** in `build_retriever()`, temporarily `print(chunks)` after
  the splitter and discuss where chunk boundaries landed.
- **Make the rewrite context-aware:** right now `rewrite` is *blind* — it only
  sees the failed query, not the chunks that came back (it does **not** use
  `state["documents"]`). Upgrade it so the rewrite can react to *what* was
  retrieved. Two edits:

  1. Add `{context}` to `rewrite_chain`'s prompt in `make_chains()`:

```python
rewrite_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "The search query below did not retrieve useful documents. Look at "
            "what WAS retrieved (CONTEXT) to see what went wrong, then rewrite "
            "the query so it targets the missing information. Output ONLY the "
            "query text.",
        ),
        ("human", "Question: {question}\nQuery that failed: {previous}\n\nCONTEXT:\n{context}"),
    ])
    | llm
)
```

  2. Pass the retrieved chunks into the node in `rewrite`:

```python
def rewrite(state: ChatState) -> dict:
    previous = state["question"]
    context = "\n\n".join(state["documents"])
    better = clean_llm_query(
        rewrite_chain.invoke(
            {"question": state["question"], "previous": previous, "context": context}
        ).content
    )
    return {"question": better, "rewrite_from": previous}
```

  **Discuss:** blind reformulation just swaps synonyms; a context-aware rewrite
  can say *"these chunks are about batteries in general, but the user wants
  warranty terms — search for 'PowerCell warranty period'."* This is closer to
  real corrective RAG. Re-run the Hauler battery warranty question and compare
  the rewritten query in the trace.

## Where to go next

This is the teaching version. For production you would typically swap
`InMemoryVectorStore` for a persistent store (e.g. Chroma), add a reranking step
after retrieval, and consider contextual chunking so each chunk's embedding
carries its surrounding context. The graph structure stays the same — you're
just upgrading the pieces inside the nodes.
