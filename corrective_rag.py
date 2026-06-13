"""
CORRECTIVE RAG with LangGraph + LangChain + Voyage embeddings + Claude
======================================================================

This is a teaching example. It builds a RAG system as a GRAPH (using LangGraph)
instead of a straight line (a LangChain chain). The point of using a graph is
that it can LOOP: if the documents it retrieves don't actually answer the
question, it rewrites the question and tries retrieving again — something a
linear chain cannot do.

THE GRAPH
---------

        START
          |
          v
      [retrieve]  <--------------------+
          |                            |
          v                            |
       [grade]   "are these docs       |
          |       relevant?"           |
          |                            |
    +-----+----------------+           |
    | relevant   not        | not      |
    |            relevant   | relevant |
    |            & tries    | & tries  |
    |            left       | used up  |
    v            v          v          |
[generate]   [rewrite]  [give_up]      |
    |            |          |          |
    |            +----------|----------+   (loop back to retrieve)
    v                       v
   END                     END

KEY LANGGRAPH CONCEPTS THIS DEMONSTRATES
----------------------------------------
1. State        - a shared dict (RAGState) that every node reads from and writes to.
2. Nodes        - plain functions that take the state and return updates to it.
3. Edges        - fixed transitions ("after retrieve, always grade").
4. Conditional  - a decision function picks the next node based on the state.
   edges          (relevant -> generate, not relevant -> rewrite, etc.)
5. Cycles       - rewrite loops back to retrieve. THIS is what graphs add over chains.

------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------
    pip install langchain langchain-anthropic langchain-voyageai langgraph

    export ANTHROPIC_API_KEY="sk-ant-..."
    export VOYAGE_API_KEY="pa-..."

    python corrective_rag.py

The documents live in the ./docs folder next to this file. Add or edit .txt
files there to change the knowledge base.
------------------------------------------------------------------------------
"""

import glob
import os
import sys
from typing import List, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHAT_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "voyage-3-large"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
MAX_ATTEMPTS = 2          # how many times we'll retry retrieval before giving up
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


# ===========================================================================
# PART 1 — INDEXING (the RAG "R": build a searchable store of document chunks)
# ===========================================================================
def build_retriever():
    """Load .txt docs, split them, embed them with Voyage, and return a retriever."""
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not paths:
        print(f"ERROR: no .txt files found in {DOCS_DIR}")
        sys.exit(1)

    # Load each file into a Document, tagging it with its filename as metadata.
    # The metadata lets us show WHICH source a retrieved chunk came from.
    raw_docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw_docs.append(
                Document(page_content=f.read(),
                         metadata={"source": os.path.basename(path)})
            )
    print(f"Loaded {len(raw_docs)} documents from {DOCS_DIR}")

    # Split into chunks. split_documents keeps each chunk's source metadata.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks.")

    # Embed (Voyage) and index. InMemoryVectorStore needs no database.
    embeddings = VoyageAIEmbeddings(model=EMBED_MODEL)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    print("Embedded and indexed all chunks.\n")

    return store.as_retriever(search_kwargs={"k": TOP_K})


# ===========================================================================
# PART 2 — THE LLM CHAINS (small LangChain pieces the graph nodes will call)
# ===========================================================================
llm = ChatAnthropic(model=CHAT_MODEL, temperature=0)

# Grader: decides whether the retrieved context can answer the question.
grader_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader. Decide if the CONTEXT contains enough information "
         "to answer the QUESTION. Reply with exactly one word: 'yes' or 'no'."),
        ("human", "QUESTION: {question}\n\nCONTEXT:\n{context}"),
    ])
    | llm
)

# Rewriter: produces a better search query when the first retrieval missed.
rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "The search query below did not retrieve useful documents. Rewrite it "
         "into a single improved search query that uses different keywords and "
         "synonyms. Reply with ONLY the rewritten query, nothing else."),
        ("human", "Original question: {question}\nQuery that failed: {previous}"),
    ])
    | llm
)

# Answerer: writes the final answer grounded ONLY in the retrieved context.
answer_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are the Helios Bikes assistant. Answer the QUESTION using ONLY the "
         "CONTEXT. If the context does not contain the answer, say you don't have "
         "that information. Be concise and friendly."),
        ("human", "QUESTION: {question}\n\nCONTEXT:\n{context}"),
    ])
    | llm
)


# ===========================================================================
# PART 3 — THE GRAPH STATE
# ===========================================================================
class RAGState(TypedDict):
    original_question: str   # what the user actually asked (never changes)
    question: str            # the query used for the current retrieval (may be rewritten)
    documents: List[str]     # text of the most recently retrieved chunks
    answer: str              # the final answer
    relevant: bool           # did the grader say the docs were relevant?
    attempts: int            # how many retrieval attempts we've made


# ===========================================================================
# PART 4 — THE NODES (each is a function: state in, partial state update out)
# ===========================================================================
def retrieve(state: RAGState) -> dict:
    """Fetch the top-k chunks for the current query."""
    docs = retriever.invoke(state["question"])
    sources = ", ".join(sorted({d.metadata.get("source", "?") for d in docs}))
    print(f"  [retrieve] query={state['question']!r}  ->  sources: {sources}")
    return {
        "documents": [d.page_content for d in docs],
        "attempts": state["attempts"] + 1,
    }


def grade(state: RAGState) -> dict:
    """Ask Claude whether the retrieved chunks can answer the original question."""
    context = "\n\n".join(state["documents"])
    verdict = grader_chain.invoke(
        {"question": state["original_question"], "context": context}
    ).content.strip().lower()
    relevant = verdict.startswith("yes")
    print(f"  [grade] relevant? {relevant}")
    return {"relevant": relevant}


def rewrite(state: RAGState) -> dict:
    """Generate a better search query and loop back to retrieve."""
    better = rewrite_chain.invoke(
        {"question": state["original_question"], "previous": state["question"]}
    ).content.strip()
    print(f"  [rewrite] new query -> {better!r}")
    return {"question": better}


def generate(state: RAGState) -> dict:
    """Produce the grounded answer from the retrieved context."""
    context = "\n\n---\n\n".join(state["documents"])
    answer = answer_chain.invoke(
        {"question": state["original_question"], "context": context}
    ).content
    print("  [generate] answer written")
    return {"answer": answer}


def give_up(state: RAGState) -> dict:
    """Fallback when retrieval keeps failing — fail gracefully, never hallucinate."""
    print("  [give_up] no relevant docs after retries")
    return {
        "answer": "I'm sorry, I couldn't find that in the Helios Bikes knowledge base."
    }


# ===========================================================================
# PART 5 — THE DECISION FUNCTION for the conditional edge after "grade"
# ===========================================================================
def decide_next(state: RAGState) -> str:
    """Route to generate (success), rewrite (retry), or give_up (out of tries)."""
    if state["relevant"]:
        return "generate"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "give_up"
    return "rewrite"


# ===========================================================================
# PART 6 — ASSEMBLE THE GRAPH
# ===========================================================================
def build_graph():
    builder = StateGraph(RAGState)

    # Register the nodes.
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_node("give_up", give_up)

    # Fixed edges.
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")

    # Conditional edge: after grading, the decide_next function picks the path.
    builder.add_conditional_edges(
        "grade",
        decide_next,
        {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
    )

    # The cycle: after rewriting, go back and retrieve again.
    builder.add_edge("rewrite", "retrieve")

    # Both terminal paths end the graph.
    builder.add_edge("generate", END)
    builder.add_edge("give_up", END)

    return builder.compile()


# ===========================================================================
# PART 7 — RUN IT
# ===========================================================================
def ask(app, question: str) -> None:
    """Run one question through the graph, printing each node as it executes."""
    print(f"\nQUESTION: {question}")
    initial_state = {
        "original_question": question,
        "question": question,
        "documents": [],
        "answer": "",
        "relevant": False,
        "attempts": 0,
    }
    final_state = app.invoke(initial_state)
    print(f"ANSWER: {final_state['answer']}")
    print("-" * 70)


def main() -> None:
    for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: missing environment variable {key}. See SETUP at top.")
            sys.exit(1)

    global retriever
    retriever = build_retriever()
    app = build_graph()

    # Uncomment to print a Mermaid diagram of the graph you can paste into
    # https://mermaid.live to visualize it:
    # print(app.get_graph().draw_mermaid())

    # Demo questions chosen to exercise every path through the graph:
    demo = [
        # straightforward, answer is in one document
        "How much does the Trailblazer cost?",
        # spans two documents (products: which battery + warranty: battery term)
        "What's the warranty on the battery in the Hauler?",
        # spans two documents (battery care + warranty) and uses different wording
        "How should I look after my bike's battery during the winter?",
        # NOT in the knowledge base -> should rewrite, then give up gracefully
        "Do you sell electric scooters?",
    ]
    for q in demo:
        ask(app, q)

    # Interactive mode.
    print("\nAsk your own questions (type 'quit' to exit):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q", ""}:
            break
        ask(app, q)


if __name__ == "__main__":
    main()
