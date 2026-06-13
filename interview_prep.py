"""
AI Agent / LangChain / LangGraph / RAG — Interview Prep Studio (Streamlit)
=========================================================================

An interactive study tool for AI-engineering interviews. It teaches concepts
four ways:

    1. LEARN     — a searchable bilingual glossary (EN + 中文)
    2. SEE IT WORK — interactive labs that run the concepts live
    3. RECALL    — flashcards + a self-grading quiz
    4. EXPLAIN   — a mock interview where Claude asks, you answer, Claude grades

Design notes
------------
- Free, no-LLM features (Glossary, Flashcards, Quiz, Chunking) always work,
  even with no API keys set.
- LLM/embedding features are GATED behind a button so you never spend money by
  accident. They need ANTHROPIC_API_KEY and/or VOYAGE_API_KEY.
- The Corrective RAG lab reuses the graph from app.py (single source of truth).

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."   # for LLM labs + mock interview
    export VOYAGE_API_KEY="pa-..."          # for embedding labs + RAG
    streamlit run interview_prep.py
"""

from __future__ import annotations

import ast
import operator
import os
import random
import uuid

import streamlit as st

import glossary
import transformer_lab

CHAT_MODEL = "claude-opus-4-8"
# The Temperature lab needs a model that still exposes the `temperature` knob;
# Opus 4.8 deprecated it. Use a model that supports sampling for that demo only.
TEMP_DEMO_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "voyage-3-large"

LANG_ENGLISH = "English"
LANG_CHINESE = "中文"
LANG_BOTH = "Both / 双语"

DEMO_LABELS = {
    glossary.DEMO_EMBEDDINGS: "🧭 Embeddings lab",
    glossary.DEMO_CHUNKING: "🔪 Chunking lab",
    glossary.DEMO_RAG: "🔁 Corrective RAG lab",
    glossary.DEMO_AGENT: "🛠️ Agent lab",
    glossary.DEMO_TRANSFORMER: "🤖 Transformer lab",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_voyage() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))


def render_term_card(term: dict, lang: str) -> None:
    """Render one glossary entry honoring the language toggle."""
    show_en = lang in (LANG_ENGLISH, LANG_BOTH)
    show_zh = lang in (LANG_CHINESE, LANG_BOTH)

    if show_en and show_zh:
        st.markdown(f"#### {term['term_en']}　·　{term['term_zh']}")
    elif show_zh:
        st.markdown(f"#### {term['term_zh']}")
    else:
        st.markdown(f"#### {term['term_en']}")

    if show_en:
        st.markdown(f"**{term['def_en']}**")
        st.markdown(f"💡 _{term['analogy_en']}_")
    if show_zh:
        st.markdown(f"**{term['def_zh']}**")
        st.markdown(f"🀄 _{term['analogy_zh']}_")

    demo = term.get("demo")
    if demo and demo in DEMO_LABELS:
        st.caption(f"See it work → {DEMO_LABELS[demo]} tab")


def _answer_concept_question(context_label: str, context_text: str | None, history: list[dict]) -> str:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system = (
        "You are a concise, friendly tutor helping someone prepare for an AI "
        f"engineering interview. They are studying: {context_label}. "
    )
    if context_text:
        system += f"Reference notes: {context_text} "
    system += (
        "Answer their questions clearly with concrete examples. Keep answers short "
        "(2-5 sentences) unless they ask for more depth. If they write in Chinese, "
        "reply in Chinese."
    )

    msgs = [SystemMessage(content=system)]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))

    llm = ChatAnthropic(model=CHAT_MODEL)
    return llm.invoke(msgs).content


def ask_ai_widget(widget_key: str, context_label: str, context_text: str | None = None) -> None:
    """A shared, collapsible mini chat. Drop into any tab to let users ask the
    LLM about the current concept/question. Gated on ANTHROPIC_API_KEY."""
    if not has_anthropic():
        st.caption("💬 Set `ANTHROPIC_API_KEY` to ask the AI about this.")
        return

    open_key = f"askopen_{widget_key}"
    hist_key = f"askhist_{widget_key}"
    input_key = f"askin_{widget_key}"

    if st.button("💬 Ask questions", key=f"askbtn_{widget_key}"):
        st.session_state[open_key] = not st.session_state.get(open_key, False)

    if not st.session_state.get(open_key):
        return

    history = st.session_state.setdefault(hist_key, [])
    with st.container(border=True):
        st.caption(f"Ask about: **{context_label}**")
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        q = st.text_input(
            "Your question",
            key=input_key,
            placeholder="e.g. How is this different from X? When would I use it?",
        )
        c1, c2, _ = st.columns([1, 1, 4])
        send = c1.button("Send", key=f"asksend_{widget_key}", type="primary")
        if c2.button("Clear", key=f"askclear_{widget_key}"):
            st.session_state[hist_key] = []
            st.session_state.pop(input_key, None)
            st.rerun()

        if send and q.strip():
            history.append({"role": "user", "content": q.strip()})
            with st.spinner("Thinking…"):
                try:
                    answer = _answer_concept_question(context_label, context_text, history)
                except Exception as exc:  # noqa: BLE001
                    answer = f"Error contacting the model: {exc}"
            history.append({"role": "assistant", "content": answer})
            st.session_state[hist_key] = history
            st.session_state.pop(input_key, None)
            st.rerun()


def safe_eval(expr: str) -> float:
    """Evaluate a basic arithmetic expression safely (no names, no calls)."""
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("Only numbers and + - * / % ** are allowed.")

    return _eval(ast.parse(expr, mode="eval").body)


# ===========================================================================
# TAB: Home
# ===========================================================================
def tab_home() -> None:
    st.subheader("Interview Prep Studio")
    st.markdown(
        "A hands-on study tool for **AI Agent / LangChain / LangGraph / RAG** "
        "interviews. Learn a concept, then watch it actually run."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Concepts", len(glossary.TERMS))
    c2.metric("Categories", len(glossary.CATEGORIES))
    c3.metric("Interactive labs", "5")
    c4.metric("Languages", "EN + 中文")

    st.markdown(
        "#### How to use this\n"
        "1. **Glossary** — read the concept + a plain-English analogy (toggle 中文 in the sidebar).\n"
        "2. **Transformer** — see attention / Q-K-V and next-token generation by hand (no key needed).\n"
        "3. **Labs** — run the idea live: embeddings, chunking, the corrective-RAG graph, a tool-using agent.\n"
        "4. **Flashcards / Quiz** — test recall.\n"
        "5. **Mock Interview** — Claude asks a question, you answer, Claude grades you.\n"
    )

    keys_ok = []
    keys_ok.append(("VOYAGE_API_KEY (embeddings, RAG)", has_voyage()))
    keys_ok.append(("ANTHROPIC_API_KEY (LLM labs, interview)", has_anthropic()))
    st.markdown("#### API keys")
    for label, ok in keys_ok:
        st.markdown(f"- {'✅' if ok else '⚠️'} `{label}` — {'set' if ok else 'not set'}")
    if not all(ok for _, ok in keys_ok):
        st.info(
            "Free features (Glossary, Flashcards, Quiz, Chunking) work without keys. "
            "Set the missing keys to unlock the LLM/embedding labs."
        )


# ===========================================================================
# TAB: Glossary
# ===========================================================================
def tab_glossary(lang: str) -> None:
    st.subheader("📚 Glossary")
    grouped = glossary.by_category()

    colf1, colf2 = st.columns([2, 3])
    chosen = colf1.multiselect(
        "Categories", glossary.CATEGORIES, default=glossary.CATEGORIES
    )
    query = colf2.text_input("Search", placeholder="e.g. cosine, agent, 余弦, 智能体")

    q = query.strip().lower()
    total = 0
    for category in glossary.CATEGORIES:
        if category not in chosen:
            continue
        terms = grouped.get(category, [])
        if q:
            terms = [
                t
                for t in terms
                if q in t["term_en"].lower()
                or q in t["term_zh"].lower()
                or q in t["def_en"].lower()
                or q in t["def_zh"].lower()
            ]
        if not terms:
            continue
        st.markdown(f"### {category}")
        for term in terms:
            total += 1
            with st.expander(f"{term['term_en']} · {term['term_zh']}"):
                render_term_card(term, lang)
                ask_ai_widget(
                    f"gloss_{term['id']}",
                    f"{term['term_en']} ({term['term_zh']})",
                    f"{term['def_en']} {term['analogy_en']}",
                )

    if total == 0:
        st.warning("No matching terms. Try a different search or category.")


# ===========================================================================
# TAB: Flashcards
# ===========================================================================
def tab_flashcards(lang: str) -> None:
    st.subheader("🃏 Flashcards")
    st.caption("Read the term, guess the meaning, then reveal. Active recall beats re-reading.")

    cat = st.selectbox("Deck", ["All"] + glossary.CATEGORIES, key="fc_cat")
    pool = glossary.TERMS if cat == "All" else [t for t in glossary.TERMS if t["category"] == cat]

    # Build / rebuild deck order if the deck changed.
    deck_key = f"fc_order_{cat}"
    if deck_key not in st.session_state:
        order = list(range(len(pool)))
        random.shuffle(order)
        st.session_state[deck_key] = order
        st.session_state.fc_pos = 0
        st.session_state.fc_revealed = False

    order = st.session_state[deck_key]
    if not order:
        st.warning("Empty deck.")
        return

    pos = st.session_state.get("fc_pos", 0) % len(order)
    term = pool[order[pos]]

    st.progress((pos + 1) / len(order), text=f"Card {pos + 1} / {len(order)}")

    with st.container(border=True):
        if lang == LANG_CHINESE:
            st.markdown(f"### {term['term_zh']}")
        elif lang == LANG_BOTH:
            st.markdown(f"### {term['term_en']} · {term['term_zh']}")
        else:
            st.markdown(f"### {term['term_en']}")
        st.caption(f"Category: {term['category']}")

        if st.session_state.get("fc_revealed"):
            render_term_card(term, lang)

    ask_ai_widget(
        f"fc_{term['id']}",
        f"{term['term_en']} ({term['term_zh']})",
        f"{term['def_en']} {term['analogy_en']}",
    )

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🔍 Reveal", use_container_width=True):
        st.session_state.fc_revealed = True
        st.rerun()
    if b2.button("⬅️ Prev", use_container_width=True):
        st.session_state.fc_pos = (pos - 1) % len(order)
        st.session_state.fc_revealed = False
        st.rerun()
    if b3.button("Next ➡️", use_container_width=True):
        st.session_state.fc_pos = (pos + 1) % len(order)
        st.session_state.fc_revealed = False
        st.rerun()
    if b4.button("🔀 Shuffle", use_container_width=True):
        random.shuffle(order)
        st.session_state[deck_key] = order
        st.session_state.fc_pos = 0
        st.session_state.fc_revealed = False
        st.rerun()


# ===========================================================================
# TAB: Quiz
# ===========================================================================
def _new_quiz_question() -> None:
    correct = random.choice(glossary.TERMS)
    distractors = random.sample([t for t in glossary.TERMS if t["id"] != correct["id"]], 3)
    options = distractors + [correct]
    random.shuffle(options)
    st.session_state.quiz_q = {
        "prompt": correct["def_en"],
        "correct_id": correct["id"],
        "options": [(t["id"], t["term_en"]) for t in options],
        "explain": correct,
    }
    st.session_state.quiz_answered = False


def tab_quiz(lang: str) -> None:
    st.subheader("❓ Quiz")
    st.caption("Match the definition to the right term. No API calls — pure recall.")

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_total = 0
    if "quiz_q" not in st.session_state:
        _new_quiz_question()

    q = st.session_state.quiz_q
    st.markdown(f"**Which term means:**")
    st.info(q["prompt"])

    answered = st.session_state.get("quiz_answered", False)
    for opt_id, opt_label in q["options"]:
        if st.button(opt_label, key=f"quiz_{opt_id}", use_container_width=True, disabled=answered):
            st.session_state.quiz_answered = True
            st.session_state.quiz_total += 1
            st.session_state.quiz_last_choice = opt_id
            if opt_id == q["correct_id"]:
                st.session_state.quiz_score += 1
            st.rerun()

    if answered:
        chosen = st.session_state.get("quiz_last_choice")
        if chosen == q["correct_id"]:
            st.success("Correct! ✅")
        else:
            right = glossary.get_term(q["correct_id"])
            st.error(f"Not quite. The answer is **{right['term_en']}** ({right['term_zh']}).")
        with st.container(border=True):
            render_term_card(q["explain"], lang)
        ask_ai_widget(
            f"quiz_{q['explain']['id']}",
            f"{q['explain']['term_en']} ({q['explain']['term_zh']})",
            f"{q['explain']['def_en']} {q['explain']['analogy_en']}",
        )
        if st.button("Next question ➡️", type="primary"):
            _new_quiz_question()
            st.rerun()

    st.caption(
        f"Score: {st.session_state.quiz_score} / {st.session_state.quiz_total}"
    )


# ===========================================================================
# TAB: Embeddings lab
# ===========================================================================
@st.cache_resource(show_spinner=False)
def get_embedder():
    from langchain_voyageai import VoyageAIEmbeddings

    return VoyageAIEmbeddings(model=EMBED_MODEL)


def cosine(a, b) -> float:
    import numpy as np

    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_breakdown(a, b):
    """Return (dot, norm_a, norm_b, cosine) computed on the full vectors."""
    import numpy as np

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot, na, nb, dot / (na * nb)


def vec_preview(vec, n: int = 8) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in vec[:n]) + f", … ] ({len(vec)} dims)"


def explain_similarity_llm(a: str, b: str, score: float) -> str:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    chain = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a friendly teacher explaining embeddings to someone prepping "
                "for an AI interview. In 3-5 short sentences: (1) what an embedding "
                "vector is, (2) how cosine similarity turns two vectors into one score "
                "(dot product divided by the two lengths), and (3) interpret THIS "
                "specific score for the two sentences (≈1 = same meaning, ≈0 = "
                "unrelated, negative = opposite). Be concrete and concise.",
            ),
            ("human", "Sentence A: {a}\nSentence B: {b}\nCosine similarity: {score:.3f}"),
        ])
        | ChatAnthropic(model=CHAT_MODEL)
    )
    return chain.invoke({"a": a, "b": b, "score": score}).content


def _embed_two_sentences() -> None:
    st.markdown(
        "Enter **two sentences**. We embed each with "
        f"`{EMBED_MODEL}`, show the actual numbers, and walk through the cosine "
        "similarity math step by step."
    )
    a = st.text_input("Sentence A", value="How do I care for my battery in winter?", key="emb_a_in")
    b = st.text_input("Sentence B", value="Tips for storing an e-bike battery in the cold.", key="emb_b_in")
    n_show = st.slider("How many vector numbers to show", 4, 20, 8, key="emb_n_show")

    if st.button("Embed & compute", type="primary", key="emb_two_btn"):
        if not a.strip() or not b.strip():
            st.error("Enter both sentences.")
            return
        with st.spinner("Embedding with Voyage…"):
            embedder = get_embedder()
            va, vb = embedder.embed_documents([a.strip(), b.strip()])
        st.session_state.emb_two = {
            "a": a.strip(),
            "b": b.strip(),
            "va": va,
            "vb": vb,
        }
        st.session_state.pop("emb_two_explain", None)

    data = st.session_state.get("emb_two")
    if not data:
        return

    va, vb = data["va"], data["vb"]
    dot, na, nb, cos = cosine_breakdown(va, vb)

    st.markdown("#### 1. The embeddings (a peek at the first numbers)")
    st.markdown(f"**A** = `{vec_preview(va, n_show)}`")
    st.markdown(f"**B** = `{vec_preview(vb, n_show)}`")
    st.caption(
        "Each sentence becomes a long list of numbers — its address on the 'meaning map'. "
        "We only show the first few; the full vector has every dimension."
    )

    st.markdown("#### 2. How cosine similarity is calculated")
    st.latex(r"\cos(A,B) = \frac{A \cdot B}{\lVert A \rVert \, \lVert B \rVert} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\,\sqrt{\sum_i b_i^2}}")

    # Show the first few term-by-term products so the dot product feels concrete.
    terms = " + ".join(f"({va[i]:+.3f})({vb[i]:+.3f})" for i in range(min(3, len(va))))
    st.markdown(
        f"- **Dot product** A·B = {terms} + … = **{dot:.4f}**  \n"
        f"  _(sum over all {len(va)} dimensions, not just the 3 shown)_\n"
        f"- **Length of A** ‖A‖ = √(Σ aᵢ²) = **{na:.4f}**\n"
        f"- **Length of B** ‖B‖ = √(Σ bᵢ²) = **{nb:.4f}**\n"
        f"- **Cosine** = {dot:.4f} / ({na:.4f} × {nb:.4f}) = **`{cos:.4f}`**"
    )

    if abs(na - 1.0) < 0.02 and abs(nb - 1.0) < 0.02:
        st.caption(
            "Note: both lengths are ≈ 1, so these vectors are already *normalized*. "
            "When that's true, cosine similarity equals the plain dot product."
        )

    st.markdown("#### 3. What the score means")
    st.progress(max(0.0, min(1.0, (cos + 1) / 2)), text=f"cosine = {cos:.4f}")
    if cos >= 0.7:
        verdict = "Very similar meaning ✅"
    elif cos >= 0.4:
        verdict = "Somewhat related"
    elif cos >= 0.1:
        verdict = "Weakly related"
    else:
        verdict = "Unrelated / different"
    st.markdown(f"**{verdict}** — 1.0 = identical meaning, 0 = unrelated, negative = opposite.")

    st.divider()
    if has_anthropic():
        if st.button("🤖 Explain this with AI", key="emb_explain_btn"):
            with st.spinner("Asking Claude…"):
                st.session_state.emb_two_explain = explain_similarity_llm(
                    data["a"], data["b"], cos
                )
        if st.session_state.get("emb_two_explain"):
            st.info(st.session_state.emb_two_explain)
    else:
        st.caption("Set `ANTHROPIC_API_KEY` to get a plain-language AI explanation here.")


def _embed_ranking() -> None:
    st.markdown(
        "Type a **query** and some **candidate** sentences. We embed them all and rank "
        "candidates by **cosine similarity** — exactly what a vector store does. Notice "
        "it matches *meaning*, not shared keywords."
    )
    query = st.text_input("Query", value="How do I take care of my battery in the cold?", key="emb_rank_q")
    candidates = st.text_area(
        "Candidates (one per line)",
        value=(
            "Store the battery indoors at about 50% charge during winter.\n"
            "The Trailblazer is our mountain bike and costs $2,600.\n"
            "Do not charge the battery below freezing temperatures.\n"
            "We ship to all 50 U.S. states within 5 to 8 business days.\n"
            "Our stores are in Portland, Seattle, and San Francisco."
        ),
        height=160,
        key="emb_rank_cands",
    )
    st.caption("Tip: the winter/battery lines share almost no keywords with the query, yet they rank top.")

    if st.button("Embed & compare", type="primary", key="emb_rank_btn"):
        lines = [c.strip() for c in candidates.splitlines() if c.strip()]
        if not query.strip() or not lines:
            st.error("Need a query and at least one candidate.")
            return
        with st.spinner("Embedding with Voyage…"):
            embedder = get_embedder()
            qvec = embedder.embed_query(query.strip())
            cvecs = embedder.embed_documents(lines)
            dim = len(qvec)
            scored = sorted(
                ((cosine(qvec, v), line) for v, line in zip(cvecs, lines)),
                key=lambda x: x[0],
                reverse=True,
            )
        st.caption(f"Vector dimensionality: **{dim}** numbers per text.")
        st.markdown("**Ranked by cosine similarity to the query:**")
        for rank, (score, line) in enumerate(scored, 1):
            st.markdown(f"**{rank}. `{score:.3f}`** — {line}")
            st.progress(max(0.0, min(1.0, (score + 1) / 2)))


def tab_embeddings() -> None:
    st.subheader("🧭 Embeddings & cosine similarity")

    if not has_voyage():
        st.warning("Set `VOYAGE_API_KEY` to run this lab (it calls the Voyage embeddings API).")
        return

    mode_two, mode_rank = st.tabs(["Two sentences (see the math)", "Query vs candidates (ranking)"])
    with mode_two:
        _embed_two_sentences()
    with mode_rank:
        _embed_ranking()

    st.divider()
    ask_ai_widget(
        "lab_embeddings",
        "Embeddings & cosine similarity",
        "An embedding is a vector capturing a text's meaning; cosine similarity scores how close two vectors are (dot product over the product of their lengths).",
    )


# ===========================================================================
# TAB: Chunking lab
# ===========================================================================
def _shared_overlap(prev: str, nxt: str, max_len: int) -> str:
    """Longest suffix of prev that is also a prefix of nxt (the visible overlap)."""
    limit = min(len(prev), len(nxt), max_len)
    for k in range(limit, 0, -1):
        if prev[-k:] == nxt[:k]:
            return nxt[:k]
    return ""


def tab_chunking() -> None:
    st.subheader("🔪 Chunking playground")
    st.markdown(
        "Splitting is the first step of RAG indexing and needs **no API key**. "
        "Adjust size and overlap and watch the chunk boundaries move. Bigger chunks = "
        "more context per chunk but fewer, coarser matches; overlap avoids cutting a "
        "fact in half."
    )

    # A single long paragraph so chunk_size actually cuts mid-section and overlap
    # becomes visible. (With short paragraphs separated by blank lines, the splitter
    # breaks on the blank lines and never needs to add overlap.)
    default_text = (
        "The Helios PowerCell battery is shared across all Helios models. Use only the "
        "charger supplied with your bike; a full charge takes about four hours. You do "
        "not need to fully drain the battery before recharging, because lithium "
        "batteries last longer with partial, frequent charges than with deep "
        "discharges. Avoid leaving the battery on the charger continuously for more "
        "than twenty-four hours. Charge and store the battery at room temperature when "
        "possible, and never charge it below freezing or above 110 degrees Fahrenheit, "
        "since charging a cold battery can cause permanent capacity loss. "
        "If you ride in cold weather, keep the battery warm indoors until just before "
        "you leave, because a cold cell delivers less range and recovers it once it "
        "warms back up. For long-term storage of a month or more, leave the battery at "
        "roughly half charge and top it up every six to eight weeks so it never sits "
        "fully empty, which can damage the cells permanently. "
        "Helios bikes use regenerative braking, so a small amount of charge is returned "
        "to the battery whenever you brake or ride downhill; this slightly extends your "
        "range on hilly routes but is not a substitute for charging from the wall. "
        "The battery management system protects against overcharging, deep discharge, "
        "and overheating, and it will shut the pack down if it detects a fault. If the "
        "status light blinks red three times, the pack is too hot or too cold to charge "
        "safely; let it return to room temperature and try again. "
        "Each PowerCell is rated for about eight hundred full charge cycles before its "
        "capacity drops to eighty percent of the original, which for most riders is "
        "several years of normal use. Replacement batteries are available from any "
        "Helios service center and must be recycled rather than thrown away, since "
        "lithium cells are hazardous waste. Never attempt to open, puncture, or repair "
        "a damaged battery yourself; contact support and stop using the bike until the "
        "pack has been inspected."
    )
    text = st.text_area("Document text", value=default_text, height=260)

    c1, c2 = st.columns(2)
    size = c1.slider("chunk_size (characters)", 80, 800, 200, step=20)
    overlap = c2.slider("chunk_overlap (characters)", 0, 200, 40, step=10)
    if overlap >= size:
        st.error("chunk_overlap must be smaller than chunk_size.")
        return

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    st.success(f"Split into **{len(chunks)}** chunk(s). Each is embedded separately in RAG.")

    st.info(
        "ℹ️ **Why overlap sometimes looks like it 'does nothing':** the splitter only "
        "adds overlap when it has to cut *inside* a section. If your text breaks neatly "
        "on blank lines / sentences shorter than `chunk_size`, each piece fits whole and "
        "no overlap is needed. Lower `chunk_size` (or raise `chunk_overlap`) to force "
        "mid-section cuts and watch the shared text appear below in **bold**.",
        icon="ℹ️",
    )

    overlaps_found = 0
    for i, chunk in enumerate(chunks, 1):
        with st.container(border=True):
            shared = ""
            if i > 1 and overlap > 0:
                shared = _shared_overlap(chunks[i - 2], chunk, overlap + 20)
            tag = f" · ↕ shares {len(shared)} chars with chunk {i - 1}" if shared else ""
            st.caption(f"Chunk {i} · {len(chunk)} chars{tag}")
            if shared:
                overlaps_found += 1
                rest = chunk[len(shared):]
                # bold the leading shared region carried over from the previous chunk
                st.markdown(f"**:orange[{shared}]**{rest}")
            else:
                st.text(chunk)

    if overlap > 0 and overlaps_found == 0 and len(chunks) > 1:
        st.warning(
            "No visible overlap with these settings — every section fit within "
            f"`chunk_size={size}`. Try `chunk_size=120` to force cuts.",
            icon="⚠️",
        )

    st.divider()
    ask_ai_widget(
        "lab_chunking",
        "Chunking (size / overlap / strategies)",
        "Splitting documents into bite-size pieces to embed; overlap avoids cutting a fact in half.",
    )


# ===========================================================================
# TAB: Corrective RAG (reuses app.py)
# ===========================================================================
def tab_rag() -> None:
    st.subheader("🔁 Corrective RAG (LangGraph)")
    st.markdown(
        "The full graph: `contextualize → retrieve → grade → (generate | rewrite ↺ | "
        "give_up)`. Watch each node fire in the trace. This reuses the graph from "
        "`app.py`."
    )

    if not (has_anthropic() and has_voyage()):
        st.warning("Set both `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` to run the RAG graph.")
        return

    import app as rag_app
    from langchain_core.messages import HumanMessage

    try:
        graph, num_chunks, num_docs = rag_app.load_app()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.caption(f"Knowledge base: {num_docs} docs → {num_chunks} chunks · top-k {rag_app.TOP_K}")

    if "rag_thread" not in st.session_state:
        st.session_state.rag_thread = str(uuid.uuid4())
        st.session_state.rag_msgs = []
        st.session_state.rag_trace = []

    cc, tc = st.columns([3, 2], gap="large")

    with cc:
        for m in st.session_state.rag_msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        question = st.text_input(
            "Ask the Helios Bikes assistant",
            key="rag_input",
            placeholder="How much does the Trailblazer cost?",
        )
        ask = st.button("Ask", type="primary", key="rag_ask")
        if st.button("New conversation", key="rag_reset"):
            st.session_state.rag_thread = str(uuid.uuid4())
            st.session_state.rag_msgs = []
            st.session_state.rag_trace = []
            st.rerun()

    with tc:
        st.markdown("**Agent trace**")
        trace_box = st.container()
        if st.session_state.rag_trace and not (ask and question.strip()):
            rag_app.render_trace(st.session_state.rag_trace, trace_box)
        elif not st.session_state.rag_trace:
            st.caption("Ask a question to watch the graph run.")
            st.info(
                "Try:\n- *What's the warranty on the battery in the Hauler?*\n"
                "- *Do you sell electric scooters?* (rewrite → give up)"
            )

    if ask and question.strip():
        st.session_state.rag_msgs.append({"role": "user", "content": question.strip()})
        config = {"configurable": {"thread_id": st.session_state.rag_thread}}
        steps: list[dict] = []
        with tc:
            live = trace_box.container()
            status = st.status("Running graph…", expanded=True)
            for event in graph.stream(
                {"messages": [HumanMessage(content=question.strip())]},
                config=config,
                stream_mode="updates",
            ):
                for node_name, update in event.items():
                    step = rag_app.format_trace_step(node_name, update)
                    steps.append(step)
                    status.write(f"{step['icon']} {step['title']}")
                    rag_app.render_trace(steps, live)
            snapshot = graph.get_state(config)
            answer = snapshot.values["messages"][-1].content
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.rag_msgs.append({"role": "assistant", "content": answer})
        st.session_state.rag_trace = steps
        st.rerun()


# ===========================================================================
# TAB: Agent (tool calling / ReAct)
# ===========================================================================
def _agent_tools():
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression, e.g. '1800 + 2600 + 3200'."""
        try:
            return str(safe_eval(expression))
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    def bike_catalog(model: str) -> str:
        """Look up specs for a Helios bike model: Dart, Trailblazer, or Hauler."""
        catalog = {
            "dart": "Dart: commuter, 38 lb, 40 mi range, $1,800.",
            "trailblazer": "Trailblazer: mountain, 52 lb, 35 mi range, $2,600.",
            "hauler": "Hauler: cargo, 68 lb, 30 mi range, $3,200.",
        }
        return catalog.get(model.strip().lower(), f"No model named '{model}'.")

    return [calculator, bike_catalog]


def tab_agent() -> None:
    st.subheader("🛠️ Tool-using agent (ReAct loop)")
    st.markdown(
        "An agent is an LLM that **acts in a loop**: it reasons, calls a tool, reads "
        "the result, and repeats until done. Below, Claude has two tools — a "
        "`calculator` and a `bike_catalog`. Watch the reason → act → observe steps."
    )

    if not has_anthropic():
        st.warning("Set `ANTHROPIC_API_KEY` to run the agent.")
        return

    st.caption("Good prompts: *What's the total price of all three Helios bikes?* · *How much heavier is the Hauler than the Dart?*")
    question = st.text_input(
        "Ask the agent",
        value="What is the combined price of the Dart, Trailblazer, and Hauler?",
        key="agent_q",
    )
    max_steps = st.slider("Max tool-loop steps", 1, 6, 4, key="agent_steps")

    if st.button("Run agent", type="primary"):
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        tools = _agent_tools()
        tool_map = {t.name: t for t in tools}
        llm = ChatAnthropic(model=CHAT_MODEL).bind_tools(tools)

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful agent. Use the provided tools to get exact "
                    "numbers instead of guessing. When you have the final answer, "
                    "reply in plain text."
                )
            ),
            HumanMessage(content=question.strip()),
        ]

        status = st.status("Agent running…", expanded=True)
        final = None
        for step in range(1, max_steps + 1):
            ai = llm.invoke(messages)
            messages.append(ai)
            if ai.tool_calls:
                if ai.content:
                    status.write(f"🧠 **Reason {step}:** {ai.content}")
                for tc in ai.tool_calls:
                    status.write(f"🔧 **Act {step}:** `{tc['name']}({tc['args']})`")
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    status.write(f"👀 **Observe {step}:** {result}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            else:
                final = ai.content
                status.write("✅ **Final answer ready**")
                break
        status.update(label="Done", state="complete", expanded=True)

        if final is None:
            st.warning("Hit the step limit before finishing. Try raising max steps.")
            final = messages[-1].content if isinstance(messages[-1].content, str) else "(no answer)"
        st.markdown("### Answer")
        st.success(final)
        st.caption(
            "This is the ReAct pattern. In real code, `create_agent` wraps this loop "
            "for you; here it's unrolled so you can see every step."
        )

    st.divider()
    ask_ai_widget(
        "lab_agent",
        "Agents, tool calling, and the ReAct loop",
        "An agent is an LLM that acts in a loop: reason, call a tool, read the result, repeat. ReAct = reason + act. create_agent wraps this loop.",
    )


# ===========================================================================
# TAB: Temperature
# ===========================================================================
def tab_temperature() -> None:
    st.subheader("🌡️ Temperature")
    st.markdown(
        "Temperature controls randomness. **0** = safe and repeatable; higher = more "
        "creative and varied. Same prompt, three temperatures, side by side."
    )
    st.caption(
        f"Interview note: newer models (like `{CHAT_MODEL}`) **deprecate** the "
        f"temperature knob, so this demo uses `{TEMP_DEMO_MODEL}`, which still "
        "supports it."
    )

    if not has_anthropic():
        st.warning("Set `ANTHROPIC_API_KEY` to run this lab.")
        return

    prompt = st.text_input("Prompt", value="Write a one-line slogan for an electric cargo bike.")
    temps = st.multiselect("Temperatures", [0.0, 0.3, 0.7, 1.0], default=[0.0, 0.7, 1.0])

    if st.button("Generate", type="primary") and prompt.strip() and temps:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        cols = st.columns(len(temps))
        for col, temp in zip(cols, sorted(temps)):
            with col:
                st.markdown(f"**temp = {temp}**")
                with st.spinner("…"):
                    llm = ChatAnthropic(model=TEMP_DEMO_MODEL, temperature=temp)
                    out = llm.invoke([HumanMessage(content=prompt.strip())]).content
                st.write(out)
        st.caption("Re-run a few times: temp 0 barely changes; high temp varies a lot.")

    st.divider()
    ask_ai_widget(
        "lab_temperature",
        "Temperature / top-p (sampling)",
        "Temperature controls output randomness: 0 is deterministic and safe, higher is more creative and varied.",
    )


# ===========================================================================
# TAB: Mock Interview
# ===========================================================================
def _mock_followup_answer(term: dict, answer: str, feedback: str, history: list[dict]) -> str:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system = (
        "You are a senior AI engineer continuing a mock interview with a candidate. "
        f"The concept under discussion is {term['term_en']} ({term['term_zh']}). "
        f"Reference notes (ground truth): {term['def_en']} {term['analogy_en']} "
        f"The candidate's original answer was: {answer or '(no answer given)'} "
        f"Your earlier written feedback was: {feedback} "
        "Now respond to the candidate's follow-up questions: be helpful, concrete, and "
        "concise (2-5 sentences), use examples, and where useful push them with a "
        "sharper follow-up question of your own. If they write in Chinese, reply in Chinese."
    )
    msgs = [SystemMessage(content=system)]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return ChatAnthropic(model=CHAT_MODEL).invoke(msgs).content


def tab_mock() -> None:
    st.subheader("🎤 Mock interview")
    st.markdown(
        "Claude plays the interviewer: it asks about a concept, you answer in your own "
        "words, and it grades you 1–5 with feedback and a model answer. This is the "
        "**LLM-as-judge** pattern — itself an interview topic."
    )

    if not has_anthropic():
        st.warning("Set `ANTHROPIC_API_KEY` to run the mock interview.")
        return

    cat = st.selectbox("Topic area", ["Any"] + glossary.CATEGORIES, key="mock_cat")

    if st.button("🎲 New question", type="primary"):
        pool = glossary.TERMS if cat == "Any" else [t for t in glossary.TERMS if t["category"] == cat]
        term = random.choice(pool)
        st.session_state.mock_term = term
        st.session_state.mock_graded = None
        st.session_state.mock_followup = []
        st.session_state.pop("mock_fu_in", None)

    term = st.session_state.get("mock_term")
    if not term:
        st.info("Click **New question** to begin.")
        return

    st.markdown("#### Interviewer")
    st.info(f"Explain **{term['term_en']}** ({term['term_zh']}). What is it, and when would you use it?")

    answer = st.text_area("Your answer", height=140, key="mock_answer")

    if st.button("Submit for grading"):
        if not answer.strip():
            st.error("Write an answer first.")
        else:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate

            judge = (
                ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are a senior AI engineer interviewer. Grade the candidate's "
                        "answer about a concept. Use the REFERENCE as ground truth. Be "
                        "encouraging but honest. Respond in this exact markdown format:\n"
                        "**Score: X/5**\n\n**What was good:** ...\n\n"
                        "**What to add:** ...\n\n**Model answer:** (2-3 sentences)",
                    ),
                    (
                        "human",
                        "CONCEPT: {term}\nREFERENCE: {reference}\n\n"
                        "CANDIDATE ANSWER: {answer}",
                    ),
                ])
                | ChatAnthropic(model=CHAT_MODEL)
            )
            with st.spinner("Interviewer is grading…"):
                feedback = judge.invoke(
                    {
                        "term": f"{term['term_en']} ({term['term_zh']})",
                        "reference": f"{term['def_en']} {term['analogy_en']}",
                        "answer": answer.strip(),
                    }
                ).content
            st.session_state.mock_graded = feedback

    if st.session_state.get("mock_graded"):
        st.markdown("#### Feedback")
        with st.container(border=True):
            st.markdown(st.session_state.mock_graded)

        st.markdown("#### Follow-up questions")
        st.caption(
            "Keep the conversation going — ask the interviewer for clarification, a "
            "concrete example, or to go deeper. It remembers your answer and its feedback."
        )
        fu = st.session_state.setdefault("mock_followup", [])
        for m in fu:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        fq = st.text_input(
            "Ask a follow-up",
            key="mock_fu_in",
            placeholder="e.g. Why wasn't that a 5? Can you give a real-world example?",
        )
        cc1, cc2, _ = st.columns([1, 1, 4])
        send = cc1.button("Send", key="mock_fu_send", type="primary")
        if cc2.button("Clear", key="mock_fu_clear"):
            st.session_state.mock_followup = []
            st.session_state.pop("mock_fu_in", None)
            st.rerun()

        if send and fq.strip():
            fu.append({"role": "user", "content": fq.strip()})
            with st.spinner("Interviewer is replying…"):
                try:
                    reply = _mock_followup_answer(
                        term,
                        st.session_state.get("mock_answer", ""),
                        st.session_state.mock_graded,
                        fu,
                    )
                except Exception as exc:  # noqa: BLE001
                    reply = f"Error contacting the model: {exc}"
            fu.append({"role": "assistant", "content": reply})
            st.session_state.mock_followup = fu
            st.session_state.pop("mock_fu_in", None)
            st.rerun()


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    st.set_page_config(page_title="AI Interview Prep Studio", page_icon="🎓", layout="wide")

    with st.sidebar:
        st.header("🎓 Interview Prep")
        st.caption("Agents · LangChain · LangGraph · RAG")
        lang = st.radio("Glossary language", [LANG_ENGLISH, LANG_CHINESE, LANG_BOTH], index=0)
        st.divider()
        st.markdown(
            f"- Chat model: `{CHAT_MODEL}`\n"
            f"- Embeddings: `{EMBED_MODEL}`"
        )
        st.caption(
            "Free tabs need no keys. Labs marked 🔒 call paid APIs and are gated "
            "behind a button."
        )

    st.title("AI Engineering Interview Prep")

    tabs = st.tabs([
        "🏠 Home",
        "📚 Glossary",
        "🃏 Flashcards",
        "❓ Quiz",
        "🤖 Transformer",
        "🧭 Embeddings 🔒",
        "🔪 Chunking",
        "🔁 Corrective RAG 🔒",
        "🛠️ Agent 🔒",
        "🌡️ Temperature 🔒",
        "🎤 Mock Interview 🔒",
    ])

    with tabs[0]:
        tab_home()
    with tabs[1]:
        tab_glossary(lang)
    with tabs[2]:
        tab_flashcards(lang)
    with tabs[3]:
        tab_quiz(lang)
    with tabs[4]:
        transformer_lab.tab_transformer(ask_ai_widget)
    with tabs[5]:
        tab_embeddings()
    with tabs[6]:
        tab_chunking()
    with tabs[7]:
        tab_rag()
    with tabs[8]:
        tab_agent()
    with tabs[9]:
        tab_temperature()
    with tabs[10]:
        tab_mock()


if __name__ == "__main__":
    main()
