"""
CONVERSATIONAL (MEMORY-BACKED) RAG CHATBOT
==========================================

This is the sequel to corrective_rag.py. That version answered each question
fresh, with no memory. THIS version remembers the conversation, so follow-ups
work:

    > How much does the Trailblazer cost?
    > What about the Hauler?          <-- understands this means "cost"
    > And how heavy is it?            <-- understands "it" = the Hauler

TWO NEW IDEAS COMPARED TO corrective_rag.py
-------------------------------------------
1. A CHECKPOINTER (LangGraph's memory). We compile the graph with
   `InMemorySaver()` and run every turn under the same `thread_id`. LangGraph
   then saves and reloads the state between turns automatically, so the
   `messages` list grows across the whole conversation.

2. A CONTEXTUALIZE step. RAG retrieves by embedding the question, but
   "what about the Hauler?" embeds badly on its own. So before retrieving, we
   ask Claude to rewrite the latest message into a STANDALONE question using the
   chat history. We retrieve with that standalone version, but still answer in
   the natural flow of the conversation.

Everything else — the retrieve -> grade -> (generate | rewrite-loop | give_up)
correction graph — is the same as before.

------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------
    pip install langchain langchain-anthropic langchain-voyageai langgraph

    export ANTHROPIC_API_KEY="sk-ant-..."
    export VOYAGE_API_KEY="pa-..."

    python chatbot_rag.py

Documents are read from the ./docs folder, same as the other scripts.
------------------------------------------------------------------------------
"""

import glob
import os
import sys
from typing import Annotated, List, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHAT_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "voyage-3-large"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
MAX_ATTEMPTS = 2
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


# ===========================================================================
# PART 1 — INDEXING (identical to the stateless version)
# ===========================================================================
def build_retriever():
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not paths:
        print(f"ERROR: no .txt files found in {DOCS_DIR}")
        sys.exit(1)

    raw_docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw_docs.append(
                Document(page_content=f.read(),
                         metadata={"source": os.path.basename(path)})
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    embeddings = VoyageAIEmbeddings(model=EMBED_MODEL)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks from {len(raw_docs)} documents.\n")
    return store.as_retriever(search_kwargs={"k": TOP_K})


# ===========================================================================
# PART 2 — LLM CHAINS
# ===========================================================================
llm = ChatAnthropic(model=CHAT_MODEL, temperature=0)

# NEW: turn a follow-up into a standalone question using the chat history.
contextualize_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation so far and a follow-up message, rewrite the "
         "follow-up as a standalone question that makes sense without the "
         "history. If it is already standalone, return it unchanged. Reply with "
         "ONLY the question text."),
        MessagesPlaceholder("history"),
        ("human", "Follow-up message: {followup}"),
    ])
    | llm
)

grader_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader. Decide if the CONTEXT contains enough information to "
         "answer the QUESTION. Reply with exactly one word: 'yes' or 'no'."),
        ("human", "QUESTION: {question}\n\nCONTEXT:\n{context}"),
    ])
    | llm
)

rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "The search query below did not retrieve useful documents. Rewrite it "
         "into a single improved search query using different keywords. Reply "
         "with ONLY the rewritten query."),
        ("human", "Question: {question}\nQuery that failed: {previous}"),
    ])
    | llm
)

# Answerer now sees the whole conversation (history) plus the retrieved context.
answer_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are the Helios Bikes assistant. Answer the user's latest message "
         "using the CONTEXT below and the conversation so far. If the context "
         "does not contain the answer, say you don't have that information. Be "
         "concise and friendly.\n\nCONTEXT:\n{context}"),
        MessagesPlaceholder("history"),
    ])
    | llm
)


# ===========================================================================
# PART 3 — STATE
# ===========================================================================
class ChatState(TypedDict):
    # add_messages is a reducer: returning new messages APPENDS them to the list
    # instead of overwriting it. Combined with the checkpointer, this is how the
    # conversation accumulates across turns.
    messages: Annotated[list, add_messages]
    question: str          # the standalone question used for retrieval this turn
    documents: List[str]
    relevant: bool
    attempts: int


# ===========================================================================
# PART 4 — NODES
# ===========================================================================
def contextualize(state: ChatState) -> dict:
    """Rewrite the latest user message into a standalone search question.

    Also resets the per-turn fields (attempts/relevant) so the correction loop
    starts fresh each turn — important, because the checkpointer would otherwise
    carry last turn's counters over.
    """
    messages = state["messages"]
    latest = messages[-1].content
    if len(messages) == 1:
        standalone = latest                      # first turn: nothing to resolve
    else:
        standalone = contextualize_chain.invoke(
            {"history": messages[:-1], "followup": latest}
        ).content.strip()
    if standalone != latest:
        print(f"  [contextualize] {latest!r} -> {standalone!r}")
    return {"question": standalone, "attempts": 0, "relevant": False}


def retrieve(state: ChatState) -> dict:
    docs = retriever.invoke(state["question"])
    sources = ", ".join(sorted({d.metadata.get("source", "?") for d in docs}))
    print(f"  [retrieve] query={state['question']!r} -> {sources}")
    return {
        "documents": [d.page_content for d in docs],
        "attempts": state["attempts"] + 1,
    }


def grade(state: ChatState) -> dict:
    context = "\n\n".join(state["documents"])
    verdict = grader_chain.invoke(
        {"question": state["question"], "context": context}
    ).content.strip().lower()
    relevant = verdict.startswith("yes")
    print(f"  [grade] relevant? {relevant}")
    return {"relevant": relevant}


def rewrite(state: ChatState) -> dict:
    better = rewrite_chain.invoke(
        {"question": state["question"], "previous": state["question"]}
    ).content.strip()
    print(f"  [rewrite] new query -> {better!r}")
    return {"question": better}


def generate(state: ChatState) -> dict:
    context = "\n\n---\n\n".join(state["documents"])
    answer = answer_chain.invoke(
        {"context": context, "history": state["messages"]}
    ).content
    # Append the assistant's reply to the conversation (add_messages reducer).
    return {"messages": [AIMessage(content=answer)]}


def give_up(state: ChatState) -> dict:
    return {"messages": [AIMessage(
        content="I'm sorry, I couldn't find that in the Helios Bikes knowledge base."
    )]}


def decide_next(state: ChatState) -> str:
    if state["relevant"]:
        return "generate"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "give_up"
    return "rewrite"


# ===========================================================================
# PART 5 — ASSEMBLE THE GRAPH (note the new contextualize node at the front)
# ===========================================================================
def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("contextualize", contextualize)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_node("give_up", give_up)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        decide_next,
        {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("give_up", END)

    # THE MEMORY: compiling with a checkpointer makes the graph save and reload
    # its state per thread_id between invocations.
    return builder.compile(checkpointer=InMemorySaver())


# ===========================================================================
# PART 6 — RUN IT
# ===========================================================================
def say(app, text: str, config: dict) -> None:
    """Send one user turn. We pass ONLY the new message; the checkpointer + the
    add_messages reducer append it to the stored history automatically."""
    print(f"\nUSER: {text}")
    result = app.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    print(f"BOT:  {result['messages'][-1].content}")
    print("-" * 70)


def main() -> None:
    for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: missing environment variable {key}. See SETUP at top.")
            sys.exit(1)

    global retriever
    retriever = build_retriever()
    app = build_graph()

    # One thread_id == one conversation. Reuse it so turns share memory.
    config = {"configurable": {"thread_id": "demo-session"}}

    # A scripted conversation that shows memory working. Notice turns 2-4 are
    # follow-ups that only make sense because the bot remembers turn 1.
    print("=" * 70)
    print("SCRIPTED DEMO (watch the [contextualize] step resolve follow-ups)")
    print("=" * 70)
    say(app, "How much does the Trailblazer cost?", config)
    say(app, "What about the Hauler?", config)          # "cost" is implied
    say(app, "And how heavy is it?", config)            # "it" = the Hauler
    say(app, "Is its battery covered by warranty?", config)  # memory + 2 docs

    # Interactive mode continues the SAME conversation (same thread_id), so it
    # remembers everything above. Type 'quit' to exit.
    print("\nKeep chatting (it remembers the conversation). Type 'quit' to exit:")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q", ""}:
            break
        say(app, text, config)


if __name__ == "__main__":
    main()
