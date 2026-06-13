"""
Helios Bikes — Conversational Corrective RAG (Streamlit)
========================================================

A teaching demo: chat with a memory-backed RAG agent and watch each graph
step live (contextualize → retrieve via embeddings → grade → generate or
rewrite loop → give up).

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."
    export VOYAGE_API_KEY="pa-..."
    streamlit run app.py

Knowledge base: ./docs/*.txt
"""

from __future__ import annotations

import glob
import os
import uuid
from typing import Annotated, List, TypedDict

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHAT_MODEL = "claude-opus-4-8"
EMBED_MODEL = "voyage-3-large"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 4
MAX_ATTEMPTS = 2
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

NODE_ICONS = {
    "contextualize": "💬",
    "retrieve": "🔍",
    "grade": "⚖️",
    "rewrite": "✏️",
    "generate": "✅",
    "give_up": "🛑",
}


def clean_llm_query(text: str) -> str:
    """Strip preambles and extra lines from LLM query outputs."""
    text = text.strip().strip('"').strip("'")
    lower = text.lower()
    for prefix in (
        "here is a rewritten search query:",
        "here's a rewritten search query:",
        "rewritten search query:",
        "rewritten query:",
        "search query:",
    ):
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if "\n" in text:
        text = next(line.strip() for line in text.splitlines() if line.strip())
    return text


# ===========================================================================
# Indexing
# ===========================================================================
def build_retriever():
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {DOCS_DIR}")

    raw_docs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw_docs.append(
                Document(
                    page_content=f.read(),
                    metadata={"source": os.path.basename(path)},
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    embeddings = VoyageAIEmbeddings(model=EMBED_MODEL)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return store.as_retriever(search_kwargs={"k": TOP_K}), len(chunks), len(raw_docs)


# ===========================================================================
# LLM chains
# ===========================================================================
def make_chains():
    llm = ChatAnthropic(model=CHAT_MODEL)

    contextualize_chain = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "Given the conversation so far and a follow-up message, rewrite the "
                "follow-up as a standalone question that makes sense without the "
                "history. If it is already standalone, return it unchanged. Reply "
                "with ONLY the question text.",
            ),
            MessagesPlaceholder("history"),
            ("human", "Follow-up message: {followup}"),
        ])
        | llm
    )

    grader_chain = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader for a Helios Bikes knowledge base (electric bicycles). "
                "The CONTEXT is one or more retrieved text chunks — they may come from "
                "different documents. Decide if the CONTEXT contains enough information "
                "to answer the QUESTION, including when the answer requires combining "
                "facts across chunks (e.g. which battery a model uses in one chunk and "
                "the warranty term in another). Reply with exactly one word: 'yes' or 'no'.",
            ),
            ("human", "QUESTION: {question}\n\nCONTEXT:\n{context}"),
        ])
        | llm
    )

    rewrite_chain = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "The search query below did not retrieve useful documents from a Helios "
                "Bikes knowledge base (electric bikes: Dart, Trailblazer, Hauler; shared "
                "PowerCell battery). Rewrite it into ONE improved search query using "
                "different keywords or synonyms. Stay on topic — do not add unrelated "
                "terms (no cars, solar, scooters, etc.). Output ONLY the query text, "
                "no labels, no explanation, no quotes.",
            ),
            ("human", "Question: {question}\nQuery that failed: {previous}"),
        ])
        | llm
    )

    answer_chain = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the Helios Bikes assistant. Answer the user's latest message "
                "using the CONTEXT below and the conversation so far. If the context "
                "does not contain the answer, say you don't have that information. Be "
                "concise and friendly.\n\nCONTEXT:\n{context}",
            ),
            MessagesPlaceholder("history"),
        ])
        | llm
    )

    return contextualize_chain, grader_chain, rewrite_chain, answer_chain


# ===========================================================================
# Graph state & nodes
# ===========================================================================
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    followup_text: str
    documents: List[str]
    retrieved_meta: List[dict]
    grade_verdict: str
    rewrite_from: str
    relevant: bool
    attempts: int


def make_graph(retriever, chains):
    contextualize_chain, grader_chain, rewrite_chain, answer_chain = chains

    def contextualize(state: ChatState) -> dict:
        messages = state["messages"]
        latest = messages[-1].content
        if len(messages) == 1:
            standalone = latest
        else:
            standalone = clean_llm_query(
                contextualize_chain.invoke(
                    {"history": messages[:-1], "followup": latest}
                ).content
            )
        return {
            "question": standalone,
            "followup_text": latest,
            "attempts": 0,
            "relevant": False,
            "rewrite_from": "",
            "retrieved_meta": [],
            "grade_verdict": "",
        }

    def retrieve(state: ChatState) -> dict:
        docs = retriever.invoke(state["question"])
        return {
            "question": state["question"],  # echoed for trace display in stream updates
            "documents": [d.page_content for d in docs],
            "retrieved_meta": [
                {
                    "source": d.metadata.get("source", "?"),
                    "preview": d.page_content[:280].replace("\n", " "),
                }
                for d in docs
            ],
            "attempts": state["attempts"] + 1,
        }

    def grade(state: ChatState) -> dict:
        context = "\n\n".join(state["documents"])
        verdict = grader_chain.invoke(
            {"question": state["question"], "context": context}
        ).content.strip().lower()
        return {"relevant": verdict.startswith("yes"), "grade_verdict": verdict}

    def rewrite(state: ChatState) -> dict:
        previous = state["question"]
        better = clean_llm_query(
            rewrite_chain.invoke(
                {"question": state["question"], "previous": previous}
            ).content
        )
        return {"question": better, "rewrite_from": previous}

    def generate(state: ChatState) -> dict:
        context = "\n\n---\n\n".join(state["documents"])
        answer = answer_chain.invoke(
            {"context": context, "history": state["messages"]}
        ).content
        return {"messages": [AIMessage(content=answer)]}

    def give_up(state: ChatState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="I'm sorry, I couldn't find that in the Helios Bikes knowledge base."
                )
            ]
        }

    def decide_next(state: ChatState) -> str:
        if state["relevant"]:
            return "generate"
        if state["attempts"] >= MAX_ATTEMPTS:
            return "give_up"
        return "rewrite"

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

    return builder.compile(checkpointer=InMemorySaver())


# ===========================================================================
# Trace formatting
# ===========================================================================
def format_trace_step(node: str, update: dict) -> dict:
    icon = NODE_ICONS.get(node, "•")

    if node == "contextualize":
        followup = update.get("followup_text", "")
        standalone = update.get("question", followup)
        if followup != standalone:
            body = (
                f"The **contextualize** node calls `contextualize_chain` "
                f"(`ChatPromptTemplate` + `ChatAnthropic`) to resolve follow-ups "
                f"using chat history.\n\n"
                f"Follow-up message: `{followup}`\n\n"
                f"→ Standalone search query: `{standalone}`"
            )
        else:
            body = (
                f"The **contextualize** node calls `contextualize_chain` only when "
                f"needed; the first message is already standalone.\n\n"
                f"Query: `{standalone}`"
            )
        return {"node": node, "icon": icon, "title": "Contextualize", "body": body}

    if node == "retrieve":
        query = update.get("question", "")
        chunks = update.get("retrieved_meta", [])
        attempt = update.get("attempts", 1)
        lines = [
            f"The **retrieve** node calls `retriever.invoke()` — LangChain embeds "
            f"the query with **{EMBED_MODEL}** (`VoyageAIEmbeddings`) and cosine-searches "
            f"`InMemoryVectorStore` (top **{TOP_K}**, attempt **{attempt}**).",
            f"\n\nSearch query: `{query}`",
            f"\n\nRetrieved **{len(chunks)}** chunk(s):",
        ]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"\n\n**{i}. {chunk['source']}**\n\n> {chunk['preview']}…")
        return {
            "node": node,
            "icon": icon,
            "title": f"Retrieve (attempt {attempt})",
            "body": "".join(lines),
        }

    if node == "grade":
        verdict = update.get("grade_verdict", "?")
        relevant = update.get("relevant", False)
        label = (
            "**relevant** — proceeding to generate"
            if relevant
            else "**not relevant** — will rewrite or give up"
        )
        return {
            "node": node,
            "icon": icon,
            "title": "Grade documents",
            "body": (
                f"The **grade** node calls a small LangChain chain named `grader_chain`: "
                f"it sends the question + retrieved chunks to Claude (`{CHAT_MODEL}`) "
                f"and asks for a one-word yes/no verdict.\n\n"
                f"Grader replied `{verdict}` → docs are {label}."
            ),
        }

    if node == "rewrite":
        prev = update.get("rewrite_from", "")
        new = update.get("question", "")
        return {
            "node": node,
            "icon": icon,
            "title": "Rewrite query",
            "body": (
                f"The **rewrite** node calls `rewrite_chain` "
                f"(`ChatPromptTemplate` + `ChatAnthropic`) after the grader said "
                f"*no*. LangGraph then loops via `add_edge(\"rewrite\", \"retrieve\")`.\n\n"
                f"`{prev}` → `{new}`"
            ),
        }

    if node == "generate":
        return {
            "node": node,
            "icon": icon,
            "title": "Generate answer",
            "body": (
                f"The **generate** node calls `answer_chain` "
                f"(`ChatPromptTemplate` + `ChatAnthropic` at `{CHAT_MODEL}`) with "
                "retrieved context and chat history."
            ),
        }

    if node == "give_up":
        return {
            "node": node,
            "icon": icon,
            "title": "Give up",
            "body": (
                f"LangGraph's `decide_next()` routed here: `attempts >= {MAX_ATTEMPTS}` "
                f"after grading *no*. No LLM call — a fixed fallback message is returned."
            ),
        }

    return {"node": node, "icon": icon, "title": node, "body": str(update)}


def render_trace(steps: list[dict], container) -> None:
    for i, step in enumerate(steps, 1):
        expanded = i == len(steps)
        with container.expander(f"{step['icon']} {i}. {step['title']}", expanded=expanded):
            st.markdown(step["body"])


# ===========================================================================
# Streamlit UI
# ===========================================================================
@st.cache_resource
def load_app():
    retriever, num_chunks, num_docs = build_retriever()
    chains = make_chains()
    graph = make_graph(retriever, chains)
    return graph, num_chunks, num_docs


def init_session():
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "chat_messages": [],
        "last_trace": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(
        page_title="Helios Bikes RAG",
        page_icon="🚲",
        layout="wide",
    )

    missing = [k for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY") if not os.environ.get(k)]
    if missing:
        st.error(
            "Missing environment variable(s): "
            + ", ".join(missing)
            + ". Set them before running:\n\n"
            "`export ANTHROPIC_API_KEY=...`  \n"
            "`export VOYAGE_API_KEY=...`"
        )
        st.stop()

    init_session()

    try:
        app, num_chunks, num_docs = load_app()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Helios Bikes RAG")
        st.caption("Conversational corrective RAG with LangGraph")
        st.markdown(
            f"- **Chat model:** `{CHAT_MODEL}`\n"
            f"- **Embeddings:** `{EMBED_MODEL}`\n"
            f"- **Knowledge base:** {num_docs} docs → {num_chunks} chunks\n"
            f"- **Top-k:** {TOP_K} · **Max retries:** {MAX_ATTEMPTS}"
        )
        st.divider()
        st.markdown(
            "**Graph:** contextualize → retrieve → grade → "
            "generate | rewrite ↺ | give up"
        )
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.session_state.last_trace = []
            st.rerun()
        st.caption(f"Session: `{st.session_state.thread_id[:8]}…`")

    st.title("🚲 Helios Bikes Assistant")
    st.caption("Ask about products, warranty, battery care, shipping, or support.")

    chat_col, trace_col = st.columns([3, 2], gap="large")

    with chat_col:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with trace_col:
        st.subheader("Agent trace")
        trace_area = st.container()
        if st.session_state.last_trace:
            render_trace(st.session_state.last_trace, trace_area)
        else:
            st.caption("Send a message to watch each graph step run live.")
            st.info(
                "Try:\n"
                "- *How much does the Trailblazer cost?*\n"
                "- *What about the Hauler?* (follow-up)\n"
                "- *Do you sell electric scooters?* (rewrite → give up)"
            )

    prompt = st.chat_input("Ask a question…")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        with chat_col:
            with st.chat_message("assistant"):
                live_trace = trace_col.empty()
                with live_trace.container():
                    st.subheader("Agent trace")
                    trace_live = st.container()
                    steps: list[dict] = []

                    status = st.status("Running graph…", expanded=True)
                    for event in app.stream(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config,
                        stream_mode="updates",
                    ):
                        for node_name, update in event.items():
                            step = format_trace_step(node_name, update)
                            steps.append(step)
                            status.write(f"{step['icon']} {step['title']}")
                            render_trace(steps, trace_live)

                    snapshot = app.get_state(config)
                    answer = snapshot.values["messages"][-1].content
                    status.update(label="Done", state="complete", expanded=False)
                    st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.last_trace = steps
        st.rerun()


if __name__ == "__main__":
    main()
