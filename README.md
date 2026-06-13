# Teaching RAG with LangGraph — Helios Bikes Example

A hands-on example for teaching Retrieval-Augmented Generation (RAG) built as a
**LangGraph** graph, using **LangChain** components, **Voyage AI** embeddings,
and **Claude** for generation.

## What's in this folder

```
helios_rag/
├── corrective_rag.py     # the application (a self-correcting RAG graph)
├── docs/                 # the knowledge base — 6 short documents
│   ├── 01_company.txt
│   ├── 02_products.txt
│   ├── 03_warranty.txt
│   ├── 04_battery_care.txt
│   ├── 05_shipping.txt
│   └── 06_support.txt
└── README.md             # this file
```

The documents describe a fictional electric-bike company, *Helios Bikes*. They
are deliberately written so that facts are **spread across documents**, which is
what makes retrieval interesting to teach.

## Setup

```bash
pip install langchain langchain-anthropic langchain-voyageai langgraph

export ANTHROPIC_API_KEY="sk-ant-..."   # from console.anthropic.com
export VOYAGE_API_KEY="pa-..."          # from dash.voyageai.com

python corrective_rag.py
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

## The five LangGraph concepts in the code

| Concept | Where in the code | What it is |
|---|---|---|
| **State** | `class RAGState` | a shared dict every node reads and updates |
| **Nodes** | `retrieve`, `grade`, `rewrite`, `generate`, `give_up` | functions: state in, update out |
| **Edges** | `add_edge(...)` | fixed "always go here next" transitions |
| **Conditional edges** | `add_conditional_edges("grade", decide_next, ...)` | branch based on the state |
| **Cycle** | `add_edge("rewrite", "retrieve")` | loop back to try again |

## What the demo questions show

When you run it, the program prints each node as it fires, so students can watch
the path the graph takes:

1. **"How much does the Trailblazer cost?"** — simple. Retrieves the products
   doc, grades it relevant, generates. Straight path, no loop.
2. **"What's the warranty on the battery in the Hauler?"** — the answer needs
   *two* documents: the catalog (which battery the Hauler uses) and the warranty
   doc (the battery term). Good for showing multi-document retrieval.
3. **"How should I look after my bike's battery during the winter?"** — phrased
   differently from the source text ("look after" vs "care", "winter storage"),
   which shows embeddings matching by *meaning*, not keywords.
4. **"Do you sell electric scooters?"** — not in the knowledge base. Watch the
   graph grade the docs *not relevant*, **rewrite** the query, retrieve again,
   and finally **give up gracefully** instead of making something up. This is the
   loop and the safety net in action.

## Suggested classroom exercises

- **Break retrieval on purpose:** set `TOP_K = 1` and ask question 2. Watch it
  miss one of the two needed documents, then trigger the rewrite loop.
- **Turn off the loop:** set `MAX_ATTEMPTS = 1` and compare behavior on the
  scooter question — it gives up immediately. This isolates what the cycle buys.
- **Visualize the graph:** uncomment the `draw_mermaid()` line and paste the
  output into https://mermaid.live.
- **Add a document:** drop a new `.txt` into `docs/` (say, a financing/payment
  plan), restart, and ask about it — no code changes needed.
- **Inspect chunking:** print `chunks` after the splitter to see how the docs
  were cut, and discuss where the chunk boundaries landed.

## Where to go next

This is the teaching version. For production you would typically swap
`InMemoryVectorStore` for a persistent store (e.g. Chroma), add a reranking step
after retrieval, and consider contextual chunking so each chunk's embedding
carries its surrounding context. The graph structure stays the same — you're
just upgrading the pieces inside the nodes.
