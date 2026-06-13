"""
Transformer / attention teaching lab (Streamlit) — pure-math, no API keys.

Split out of interview_prep.py to keep that file manageable. Everything here is
a deterministic *toy*: random (but reproducible) embeddings and one attention
head, used to make the mechanism visible — not a trained language model.

Public entry point:
    tab_transformer(ask_ai_widget)   # ask_ai_widget is injected to avoid a
                                     # circular import with interview_prep.py
"""

from __future__ import annotations

from typing import Callable

import streamlit as st


# ---------------------------------------------------------------------------
# Toy math (deterministic, untrained)
# ---------------------------------------------------------------------------
def _softmax(x, temp: float = 1.0):
    import numpy as np

    x = np.asarray(x, dtype=float) / max(temp, 1e-9)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _toy_embed(token: str, d_model: int, seed: int = 0):
    """Deterministic pseudo-embedding for a token (stable across reruns)."""
    import hashlib

    import numpy as np

    out = np.zeros(d_model)
    for i in range(d_model):
        h = hashlib.md5(f"{seed}:{token.lower()}:{i}".encode()).hexdigest()
        out[i] = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0  # in [-1, 1]
    return out


def _positional(n: int, d_model: int):
    import numpy as np

    pos = np.arange(n)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((n, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe * 0.5


def _run_toy_attention(tokens, d_model: int, d_k: int, seed: int, use_pos: bool, causal: bool = False):
    """One head of self-attention on toy embeddings. Returns everything for display.

    If `causal` is True, applies a decoder mask so each token can only attend to
    itself and earlier tokens (no peeking at the future).
    """
    import numpy as np

    X = np.vstack([_toy_embed(t, d_model, seed) for t in tokens])
    if use_pos:
        X = X + _positional(len(tokens), d_model)

    rng = np.random.default_rng(seed + 1)
    Wq = rng.standard_normal((d_model, d_k)) / np.sqrt(d_model)
    Wk = rng.standard_normal((d_model, d_k)) / np.sqrt(d_model)
    Wv = rng.standard_normal((d_model, d_k)) / np.sqrt(d_model)
    Wo = rng.standard_normal((d_k, d_model)) / np.sqrt(d_k)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = (Q @ K.T) / np.sqrt(d_k)
    if causal:
        n = len(tokens)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # True above the diagonal = future
        scores = np.where(mask, -np.inf, scores)
    weights = np.vstack([_softmax(row) for row in scores])
    context = weights @ V          # (n, d_k)
    out = context @ Wo             # (n, d_model) — back to model width
    return {
        "X": X, "Q": Q, "K": K, "V": V,
        "Wq": Wq, "Wk": Wk, "Wv": Wv,
        "scores": scores, "weights": weights, "context": context, "out": out,
    }


# Toy vocabulary for the next-token lab (~200 common words, deduped at use).
BASE_VOCAB = [
    # function words
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because",
    "while", "when", "where", "which", "that", "this", "these", "those", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "into", "over", "under",
    "up", "down", "out", "off", "about", "as", "than", "not", "no", "yes",
    "it", "its", "they", "them", "he", "she", "we", "you", "i", "me",
    "my", "your", "our", "his", "her", "their", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "very", "more", "most",
    # nouns
    "cat", "dog", "bird", "fish", "tree", "house", "car", "bike", "battery", "charger",
    "road", "hill", "river", "mountain", "city", "store", "world", "time", "day", "night",
    "morning", "evening", "sun", "moon", "sky", "rain", "snow", "wind", "fire", "water",
    "food", "book", "word", "sentence", "model", "token", "vector", "data", "code", "page",
    "person", "child", "friend", "family", "hand", "eye", "heart", "mind", "voice", "door",
    # verbs
    "run", "ran", "walk", "walked", "jump", "sat", "sit", "stand", "sleep", "slept",
    "eat", "ate", "drink", "go", "went", "come", "came", "see", "saw", "look",
    "make", "made", "take", "took", "give", "gave", "find", "found", "think", "thought",
    "know", "knew", "want", "need", "use", "ride", "rode", "drive", "fly", "fall",
    "charge", "learn", "teach", "read", "write", "speak", "listen", "build", "break", "open",
    # adjectives / adverbs
    "big", "small", "fast", "slow", "good", "bad", "happy", "sad", "tired", "loud",
    "soft", "hard", "warm", "cold", "hot", "new", "old", "long", "short", "high",
    "low", "bright", "dark", "quick", "quiet", "strong", "weak", "full", "empty", "ready",
    "always", "never", "often", "soon", "later", "here", "there", "now", "again", "well",
    # misc / control
    "one", "two", "three", "many", "few", "some", "any", "all", "every", "<end>",
]


# ---------------------------------------------------------------------------
# Sub-tabs
# ---------------------------------------------------------------------------
def _tf_big_picture() -> None:
    st.markdown(
        "A **Transformer** is the engine inside GPT / Claude. It never \"thinks\" about a "
        "whole word at once — it works one **token** at a time, and its only real trick is "
        "**attention**: letting every token look at the others and borrow meaning from the "
        "relevant ones."
    )

    st.markdown("#### The pipeline, end to end")
    st.markdown(
        "1. **Tokenize** — text → tokens (sub-word pieces).\n"
        "2. **Embed** — each token → a vector (its meaning).\n"
        "3. **+ Positional encoding** — add 'seat numbers' so order matters.\n"
        "4. **Attention blocks (×N)** — tokens mix info via Q/K/V self-attention, "
        "then a small feed-forward network refines each token. Stack dozens of these.\n"
        "5. **Unembedding → logits** — the last token's final vector is scored against "
        "every word in the vocabulary.\n"
        "6. **Softmax → sample** — logits become probabilities; pick the next token.\n"
        "7. **Loop** — append that token and repeat (this is *autoregression*)."
    )
    st.caption(
        "Steps 4–6 are what the two playgrounds let you run by hand: 'Self-attention' is "
        "step 4, 'Next-token' is steps 5–7."
    )

    st.markdown("#### The one equation to remember")
    st.latex(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")
    st.markdown(
        "- **Q (Query)** — what *this* token is looking for.\n"
        "- **K (Key)** — what each token *offers* (its label).\n"
        "- **V (Value)** — the actual content each token will hand over.\n"
        "- Each token's vector `x` is projected three ways: `Q = x·W_Q`, `K = x·W_K`, `V = x·W_V`.\n"
        "- `Q·Kᵀ` scores every query against every key; `√dₖ` keeps the numbers stable; "
        "**softmax** turns scores into weights that sum to 1; multiplying by **V** blends the content."
    )

    st.markdown("#### Self-attention vs cross-attention")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Self-attention**  \n"
            "Q, K, V all come from the **same** sequence. Tokens look at *each other*.  \n"
            "_Used in: GPT/Claude decoder layers — re-reading your own text._"
        )
        st.latex(r"Q,K,V \leftarrow \text{same sequence } X")
    with c2:
        st.markdown(
            "**Cross-attention**  \n"
            "Q comes from sequence A; K, V come from a **different** sequence B.  \n"
            "_Used in: translation (decoder→encoder), and conceptually in RAG "
            "(your question attends to retrieved docs)._"
        )
        st.latex(r"Q \leftarrow A,\quad K,V \leftarrow B")

    st.markdown("#### Encoder vs Decoder")
    st.markdown(
        "Transformers come in two halves. You can use one, the other, or both:"
    )
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            "**Encoder** — *the reader*  \n"
            "Reads the **whole** input at once and builds a rich understanding of it. "
            "Every token can look both **left and right** (bidirectional). It does **not** "
            "generate text; it outputs vectors that *represent meaning*.  \n"
            "_Used for: embeddings, search, classification (e.g. BERT, the Voyage "
            "embedding model in the RAG lab)._"
        )
    with e2:
        st.markdown(
            "**Decoder** — *the writer*  \n"
            "Generates text **one token at a time**, and each token may only look at tokens "
            "**before** it (causal / masked attention — no peeking at the future). This is "
            "the next-token loop.  \n"
            "_Used for: chat & generation (e.g. GPT, Claude, Llama)._"
        )
    st.markdown(
        "**Three ways to wire them up:**\n"
        "- **Encoder-only** (BERT, embedding models) → understand / encode text into vectors.\n"
        "- **Decoder-only** (GPT, Claude) → generate text; this is what almost every modern "
        "chat LLM is.\n"
        "- **Encoder–decoder** (T5, original translation Transformer) → the encoder reads the "
        "source, the **decoder writes the output while using cross-attention to look back at "
        "the encoder's vectors**. That's exactly where cross-attention lives."
    )
    st.caption(
        "Tie-in: the embedding model in the Embeddings/RAG labs is **encoder-style** "
        "(text → meaning vectors). The Next-token playground here is **decoder-style** "
        "(generate the next token, looking only at the past)."
    )

    st.info(
        "Interview soundbite: *\"Attention is a soft, weighted lookup. Each token builds a "
        "query, matches it against every token's key, and pulls a weighted mix of their "
        "values. Self-attention queries the same sequence; cross-attention queries a "
        "different one. Encoders read bidirectionally to understand; decoders generate "
        "left-to-right with causal masking; encoder–decoder models connect the two with "
        "cross-attention.\"*"
    )


def _vec_str(v) -> str:
    return "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"


def _brow(v) -> str:
    """LaTeX 1×n row matrix."""
    return r"\begin{bmatrix}" + " & ".join(f"{x:+.2f}" for x in v) + r"\end{bmatrix}"


def _bmatrix(M) -> str:
    """LaTeX matrix from a 2-D array."""
    rows = [" & ".join(f"{x:+.2f}" for x in row) for row in M]
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


def _tf_self_attention_lab() -> None:
    import numpy as np
    import pandas as pd

    st.markdown(
        "Let's do attention **by hand** with tiny numbers. Each word asks a question "
        "(**Query**), wears a name-tag (**Key**), and carries info to share (**Value**). "
        "To decide how much word A listens to word B, we check how well A's *question* "
        "matches B's *name-tag*."
    )

    st.markdown("#### The flow, as a picture")
    st.caption("One input X fans out into Q, K, V, which recombine into the output. The steps below run this with real numbers.")
    st.graphviz_chart(
        """
        digraph attention {
            rankdir=TB;
            bgcolor="transparent";
            node [fontname="Helvetica", fontsize=11, style="filled", color="#888888"];
            edge [fontname="Helvetica", fontsize=10, color="#888888"];

            X [label="X — input embeddings\\n(one row per word)", shape=box, fillcolor="#e3f2fd"];
            Q [label="Q = X · W_Q\\n(queries)", shape=box, fillcolor="#fff3e0"];
            K [label="K = X · W_K\\n(keys)",    shape=box, fillcolor="#fff3e0"];
            V [label="V = X · W_V\\n(values)",  shape=box, fillcolor="#fff3e0"];
            S [label="match Q with K\\n(scaled dot-product)", shape=box, fillcolor="#f3e5f5"];
            W [label="softmax(each row)\\nattention weights\\n(row sums to 1)", shape=box, fillcolor="#f3e5f5"];
            O [label="output = weights · V\\n(a blend of the values)", shape=box, fillcolor="#e8f5e9"];

            X -> Q; X -> K; X -> V;
            Q -> S [label=" match"];
            K -> S [label=" match"];
            S -> W;
            W -> O [label=" weights"];
            V -> O [label=" content"];
        }
        """
    )

    text = st.text_input("Words (keep it short so the tables stay tiny)", value="the cat sat", key="tf_sa_text")
    c1, c2, c3 = st.columns(3)
    d_model = c1.slider("Numbers per word (embedding size)", 2, 6, 4, key="tf_sa_dm")
    d_k = c2.slider("Numbers per Q/K/V", 2, 4, 3, key="tf_sa_dk")
    seed = c3.slider("Random seed", 0, 20, 1, key="tf_sa_seed")
    use_pos = st.checkbox("Add positional encoding (so word order matters)", value=False, key="tf_sa_pos")

    tokens = [t for t in text.split() if t][:5]
    if len(tokens) < 2:
        st.info("Type at least two words.")
        return

    r = _run_toy_attention(tokens, d_model, d_k, seed, use_pos)

    # ---- Step 0: input tokens and their embeddings -----------------------
    st.markdown("### Step 0 · Turn each word into numbers")
    st.markdown(
        "A computer can't read letters, only numbers. So each word becomes a short list of "
        f"numbers called its **embedding**. Here every word is **{d_model} numbers**:"
    )
    x_df = pd.DataFrame(np.round(r["X"], 2), index=tokens, columns=[f"e{i}" for i in range(d_model)])
    st.dataframe(x_df, use_container_width=True)
    st.caption("Each row is one word. (These numbers are random toys, but a real model *learns* them.)")

    # ---- Step 1: the weight grids ---------------------------------------
    st.markdown("### Step 1 · The model's 3 weight grids: W_Q, W_K, W_V")
    st.markdown(
        "To make 3 versions of each word, the model keeps three fixed grids of numbers — its "
        "**weight matrices**. These are the *only* things a real model **learns**; here they're "
        f"random. Each is **{d_model}×{d_k}** (embedding size → Q/K/V size)."
    )
    wcols = st.columns(3)
    for col_box, name, lbl in zip(wcols, ("Wq", "Wk", "Wv"), ("W_Q (→ Query)", "W_K (→ Key)", "W_V (→ Value)")):
        with col_box:
            st.markdown(f"**{lbl}**")
            st.dataframe(pd.DataFrame(np.round(r[name], 2),
                                      index=[f"e{i}" for i in range(d_model)],
                                      columns=[f"{name[1].lower()}{j}" for j in range(d_k)]),
                         use_container_width=True)

    # ---- Step 2: use X to compute Q, K, V -------------------------------
    st.markdown("### Step 2 · Multiply X by each grid to get Q, K, V")
    st.markdown(
        "Now take a word's embedding from Step 0 (a **row of X**) and multiply it by a grid. "
        "`X · W_Q = Q`, `X · W_K = K`, `X · W_V = V`. Pick a word and pick which one to build:"
    )
    cwi, cwh = st.columns([2, 3])
    wi = cwi.selectbox("Word", range(len(tokens)), format_func=lambda i: tokens[i], key="tf_sa_calc_word")
    which = cwh.radio("Build its…", ["Q", "K", "V"], horizontal=True, key="tf_sa_calc_which")
    Wname = {"Q": "Wq", "K": "Wk", "V": "Wv"}[which]
    letter = which.lower()
    emb = r["X"][wi]
    W = r[Wname]
    out_vec = r[which][wi]

    # The whole operation as a labelled matrix picture.
    st.latex(
        rf"\underbrace{{{_brow(emb)}}}_{{\text{{{tokens[wi]}'s row of X}}\,(1\times{d_model})}}"
        rf"\;\times\;"
        rf"\underbrace{{{_bmatrix(W)}}}_{{W_{which}\,({d_model}\times{d_k})}}"
        rf"\;=\;"
        rf"\underbrace{{{_brow(out_vec)}}}_{{{which}\text{{ row}}\,(1\times{d_k})}}"
    )

    # Then zoom into ONE output number so it isn't a wall of equations.
    j = st.radio("Zoom into one output number:", list(range(d_k)),
                 format_func=lambda j: f"{letter}{j}", horizontal=True, key="tf_sa_calc_j")
    col = W[:, j]
    terms = " + ".join(f"({emb[i]:+.2f})({col[i]:+.2f})" for i in range(d_model))
    st.latex(rf"{letter}_{j} \;=\; {terms} \;=\; {float(np.dot(emb, col)):+.2f}")
    st.caption(
        f"`{letter}{j}` is the dot product of **{tokens[wi]}**'s embedding with **column {j}** "
        f"of W_{which}. Repeat for each column to fill the whole {which} row."
    )

    # ---- Step 2b: the resulting Q/K/V for every word -------------------
    st.markdown("#### Do that for every word → the full Q, K, V tables")
    cQ, cK, cV = st.columns(3)
    with cQ:
        st.markdown("**Q — Query** *(looking for)*")
        st.dataframe(pd.DataFrame(np.round(r["Q"], 2), index=tokens,
                                  columns=[f"q{i}" for i in range(d_k)]), use_container_width=True)
    with cK:
        st.markdown("**K — Key** *(label)*")
        st.dataframe(pd.DataFrame(np.round(r["K"], 2), index=tokens,
                                  columns=[f"k{i}" for i in range(d_k)]), use_container_width=True)
    with cV:
        st.markdown("**V — Value** *(info to share)*")
        st.dataframe(pd.DataFrame(np.round(r["V"], 2), index=tokens,
                                  columns=[f"v{i}" for i in range(d_k)]), use_container_width=True)

    # ---- Step 3: one concrete score (the dot product) -------------------
    st.markdown("### Step 3 · How much should one word listen to another?")
    st.markdown("Pick two words. We multiply the first word's **Query** by the second word's **Key**, number by number, then add it all up. Bigger total = pay more attention.")
    cc1, cc2 = st.columns(2)
    qi = cc1.selectbox("Word doing the looking (uses its Query)", range(len(tokens)),
                       format_func=lambda i: tokens[i], key="tf_sa_qi")
    kj = cc2.selectbox("Word being looked at (uses its Key)", range(len(tokens)),
                       index=min(1, len(tokens) - 1), format_func=lambda i: tokens[i], key="tf_sa_kj")

    qv, kv = r["Q"][qi], r["K"][kj]
    st.markdown(f"- **{tokens[qi]}**'s Query = `{_vec_str(qv)}`")
    st.markdown(f"- **{tokens[kj]}**'s Key  = `{_vec_str(kv)}`")
    prod_terms = " + ".join(f"({qv[i]:+.2f})\\times({kv[i]:+.2f})" for i in range(d_k))
    raw = float(np.dot(qv, kv))
    st.latex(rf"\text{{score}} = {prod_terms} = {raw:+.2f}")
    st.latex(
        rf"\text{{scaled score}} = \frac{{\text{{score}}}}{{\sqrt{{d_k}}}} "
        rf"= \frac{{{raw:+.2f}}}{{\sqrt{{{d_k}}}}} = \frac{{{raw:+.2f}}}{{{np.sqrt(d_k):.2f}}} = {raw/np.sqrt(d_k):+.2f}"
    )
    st.info(
        f"**Where does √{d_k} come from?** It's **√dₖ**, where **dₖ = {d_k}** is the length of "
        f"each Q and K vector (the *\"Numbers per Q/K/V\"* slider — you set it to {d_k}, so we "
        f"divide by √{d_k} ≈ {np.sqrt(d_k):.2f}).\n\n"
        "**Why divide at all?** A dot product adds up dₖ separate products. The more numbers "
        "you add, the bigger the totals tend to grow (they grow ~ with √dₖ). If the scores get "
        "too large, softmax becomes razor-sharp — it puts ~100% on one word and ~0% on the "
        "rest, which kills learning. Dividing by √dₖ cancels that growth and keeps the scores "
        "in a sensible range. Try the slider: dₖ=2 → divide by √2≈1.41, dₖ=4 → divide by √4=2."
    )

    # ---- Step 4: full score matrix (+ optional causal mask) ------------
    st.markdown("### Step 4 · Do that for every pair → a score grid")
    st.caption("Row = the word looking. Column = the word being looked at. (This is Step 3 repeated for all pairs.)")

    causal = st.checkbox("🚫 Apply causal mask (decoder: each word can't look ahead)",
                         value=False, key="tf_sa_causal")
    # Re-run with the mask so the grid + all later steps reflect it.
    r = _run_toy_attention(tokens, d_model, d_k, seed, use_pos, causal=causal)

    score_show = pd.DataFrame(np.round(r["scores"], 2), index=tokens, columns=tokens)
    if causal:
        score_show = score_show.replace(-np.inf, "−∞ (masked)")
    st.dataframe(score_show, use_container_width=True)
    if causal:
        st.warning(
            "🚫 **Causal mask is ON (decoder mode).** Every cell *above the diagonal* is a word "
            "looking at a **future** word — not allowed when generating text. We set those to "
            "**−∞**, so after softmax they become **0%**. Each word can only attend to itself "
            "and the words before it. (Turn the checkbox off to see full, bidirectional / "
            "encoder-style attention.)"
        )
    else:
        st.caption(
            "ℹ️ No mask: every word sees every other word (bidirectional, encoder-style). "
            "Turn on **Causal mask** above to hide future words (decoder-style, like GPT/Claude)."
        )

    st.markdown("### Step 5 · Turn each row into percentages (softmax)")
    st.markdown(
        "Scores can be any number, so we squish each **row** into nice percentages that add "
        "up to **100%**. That's the **softmax**. Now each number says *\"how much of my "
        "attention goes to this word.\"*"
    )
    st.latex(r"\text{softmax}(s)_j = \frac{e^{s_j}}{\sum_{m} e^{s_m}}")
    st.caption("Raise e to each score (makes everything positive, big scores stand out), then divide by the total so the row sums to 1.")

    # Worked example with the real numbers from one row.
    def _softmax_worked(row_i: int) -> None:
        srow = r["scores"][row_i]
        denom = float(np.sum(np.exp(srow - srow.max())) * np.exp(srow.max()))
        num_terms = " + ".join(f"e^{{{srow[m]:+.2f}}}" for m in range(len(tokens)))
        st.markdown(f"**Worked example — row for '{tokens[row_i]}'** (scores = `{_vec_str(srow)}`):")
        st.latex(rf"\text{{denominator}} = {num_terms} = {denom:.2f}")
        for j in range(len(tokens)):
            st.latex(
                rf"\text{{weight}}_{{{tokens[row_i]}\to {tokens[j]}}} = "
                rf"\frac{{e^{{{srow[j]:+.2f}}}}}{{{denom:.2f}}} = "
                rf"{r['weights'][row_i][j]:.2f}\;=\;{r['weights'][row_i][j]*100:.0f}\%"
            )

    _softmax_worked(0)
    if len(tokens) > 1:
        st.markdown("---")
        _softmax_worked(len(tokens) - 1)

    st.markdown("**The whole grid as percentages:**")
    w_df = pd.DataFrame(np.round(r["weights"] * 100, 0).astype(int), index=tokens, columns=tokens)
    st.dataframe(w_df.style.format("{:d}%"), use_container_width=True)

    # ---- Step 6: pick one word, show the blend --------------------------
    st.markdown("### Step 6 · Mix the Values using those percentages")
    qtok = st.selectbox("Show the full result for this word", range(len(tokens)),
                        format_func=lambda i: tokens[i], key="tf_sa_qtok")
    wrow = r["weights"][qtok]
    order = np.argsort(wrow)[::-1]
    st.markdown(f"**'{tokens[qtok]}'** splits its attention like this:")
    for j in order:
        st.markdown(f"- **{wrow[j]*100:.0f}%** on **{tokens[j]}** (takes that much of {tokens[j]}'s Value)")
        st.progress(float(wrow[j]))
    top = tokens[int(order[0])]
    st.success(f"So **'{tokens[qtok]}'** mostly listens to **'{top}'**. The output for "
               f"'{tokens[qtok]}' is a blend of all the Values, weighted by these percentages.")
    st.caption(
        f"Output for '{tokens[qtok]}' = " +
        " + ".join(f"{wrow[j]*100:.0f}%·V({tokens[j]})" for j in range(len(tokens))) +
        f" = `{_vec_str(np.round(r['context'][qtok], 2))}`"
    )
    st.info(
        "Try the **seed** slider (different random weights = a different 'attention head' "
        "that cares about different things), or turn on positional encoding to see word "
        "order change who listens to whom."
    )

    # ---- Step 7: from V-blend to the next token (with numbers) ----------
    st.markdown("### Step 7 · From the V-blend to the next word")

    last = tokens[-1]
    h = r["out"][-1]
    st.markdown(
        f"To predict the word *after* **'{last}'**, we take **'{last}'**'s output vector from "
        f"Step 6 (run through a few more layers) — call it `h`. This one vector carries "
        "everything the model knows so far:"
    )
    st.latex(rf"h_{{\text{{{last}}}}} = {_brow(np.round(h, 2))}")

    # The vocabulary we'll score against.
    cand = list(dict.fromkeys(tokens + ["mat", "dog", "ran", "slept"]))[:7]
    E = np.vstack([_toy_embed(w, len(h), seed) for w in cand])
    logits = E @ h
    probs = _softmax(logits)
    order = np.argsort(probs)[::-1]

    st.markdown(
        f"**The idea:** every word in the vocabulary also has an embedding. We measure how well "
        f"`h` *matches* each word with a **dot product** — that's the word's **logit** (raw "
        f"score). Big match = the model thinks that word likely comes after **'{last}'**."
    )

    st.markdown("**① One word, step by step.** Pick a candidate to see the dot product:")
    ci = st.selectbox("Candidate next word", range(len(cand)),
                      format_func=lambda i: cand[i], key="tf_sa_nextword")
    emb_c = E[ci]
    terms = " + ".join(f"({h[i]:+.2f})({emb_c[i]:+.2f})" for i in range(len(h)))
    st.latex(rf"\text{{logit}}(\text{{{cand[ci]}}}) = h \cdot \text{{emb}}(\text{{{cand[ci]}}}) = {terms} = {logits[ci]:+.2f}")

    st.markdown("**② Do that for the whole vocabulary** → one logit per word:")
    st.latex(r"\text{logits} = " + _brow(np.round(logits, 2)) + r"\quad(\text{one per candidate})")

    st.markdown("**③ Softmax the logits → a probability for every word** (same softmax as Step 5):")
    denom = float(np.sum(np.exp(logits - logits.max())) * np.exp(logits.max()))
    st.latex(
        rf"P(\text{{{cand[ci]}}}) = \frac{{e^{{{logits[ci]:+.2f}}}}}{{\sum_w e^{{\text{{logit}}(w)}}}} "
        rf"= \frac{{{np.exp(logits[ci]):.2f}}}{{{denom:.2f}}} = {probs[ci]*100:.0f}\%"
    )

    tbl = pd.DataFrame(
        {
            "candidate word": [cand[i] for i in order],
            "logit = h·emb": [f"{logits[i]:+.2f}" for i in order],
            "probability": [f"{probs[i]*100:.1f}%" for i in order],
        }
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    for i in order:
        st.progress(float(probs[i]), text=f"{cand[i]} — {probs[i]*100:.0f}%")

    winner = cand[int(order[0])]
    st.success(
        f"All probabilities add to 100%. The highest wins → next word = **'{winner}'**. "
        f"Append it (`{' '.join(tokens)} {winner}`) and repeat from Step 0. That loop is how "
        "an LLM writes one word at a time."
    )
    st.caption(
        "Numbers are random toys — the **\"Next-token generation\"** sub-tab does this exact "
        "thing with a ~200-word vocab and lets you generate word by word."
    )


def _tf_next_token_lab() -> None:
    import numpy as np

    st.markdown(
        "Watch the **autoregressive loop**: the model turns its final vector into scores "
        "over a vocabulary (**logits**), softmax makes them probabilities, one token is "
        "chosen, appended, and the whole thing repeats."
    )
    st.warning(
        "⚠️ **This is NOT a trained model.** No language model is loaded. The embeddings "
        "and attention weights are *random* (deterministic, but never trained), so the "
        "words it picks are essentially arbitrary. What's real and worth studying is the "
        "**mechanism**: hidden vector → logits → softmax → sample → loop. A real LLM does "
        "exactly these steps — it just has *learned* weights and a ~50k–200k token "
        "vocabulary instead of our ~200-word toy one.",
        icon="⚠️",
    )

    d_model, d_k, seed = 8, 6, 3

    if "tf_seq" not in st.session_state:
        st.session_state.tf_seq = ["the", "cat"]

    c1, c2 = st.columns([3, 1])
    start = c1.text_input("Starting tokens", value=" ".join(st.session_state.tf_seq), key="tf_nt_start")
    if c2.button("Reset to this", key="tf_nt_reset"):
        st.session_state.tf_seq = [t for t in start.split() if t] or ["the"]
        st.rerun()

    seq = st.session_state.tf_seq
    vocab = list(dict.fromkeys([w.lower() for w in seq] + BASE_VOCAB))

    # Build hidden state for the last position via one attention head.
    r = _run_toy_attention(seq, d_model, d_k, seed, use_pos=True, causal=True)
    h = r["out"][-1]  # final vector for the last token (d_model)

    # Unembedding: score every vocab word by dot product with the hidden state.
    E = np.vstack([_toy_embed(w, d_model, seed) for w in vocab])
    logits = E @ h

    temp = st.slider("Temperature (output randomness)", 0.1, 2.0, 0.8, 0.1, key="tf_nt_temp")
    probs = _softmax(logits, temp)

    st.markdown(f"**Current sequence:** `{' '.join(seq)}`")
    st.caption(f"The last token's vector `h` (d_model={d_model}) is compared to all {len(vocab)} vocab embeddings.")

    with st.expander("Show the raw Q / K / V and the hidden state behind this"):
        import pandas as pd

        st.caption("Same self-attention as the Q/K/V tab, run on your sequence (with causal positional encoding).")
        cQ, cK, cV = st.columns(3)
        with cQ:
            st.markdown("**Q**")
            st.dataframe(pd.DataFrame(np.round(r["Q"], 2), index=seq,
                                      columns=[f"q{i}" for i in range(d_k)]), use_container_width=True)
        with cK:
            st.markdown("**K**")
            st.dataframe(pd.DataFrame(np.round(r["K"], 2), index=seq,
                                      columns=[f"k{i}" for i in range(d_k)]), use_container_width=True)
        with cV:
            st.markdown("**V**")
            st.dataframe(pd.DataFrame(np.round(r["V"], 2), index=seq,
                                      columns=[f"v{i}" for i in range(d_k)]), use_container_width=True)
        st.markdown("**Attention output per token** (the blended Values, back at width d_model):")
        st.dataframe(pd.DataFrame(np.round(r["out"], 2), index=seq,
                                  columns=[f"o{i}" for i in range(d_model)]), use_container_width=True)
        st.markdown(
            f"**Hidden state `h`** = the last row (for **'{seq[-1]}'**) = `{_vec_str(np.round(h, 2))}`  \n"
            "This `h` is what we dot against every vocab embedding to get the logits below."
        )

    order = np.argsort(probs)[::-1][:8]
    st.markdown("#### Next-token probabilities (top 8)")
    for idx in order:
        st.markdown(f"**{vocab[idx]}** — `{probs[idx]*100:4.1f}%`  (logit {logits[idx]:+.2f})")
        st.progress(float(probs[idx]))

    greedy = vocab[int(order[0])]
    b1, b2, b3 = st.columns(3)
    if b1.button(f"➡️ Greedy: add '{greedy}'", type="primary", key="tf_nt_greedy"):
        st.session_state.tf_seq = seq + [greedy]
        st.rerun()
    if b2.button("🎲 Sample (use temperature)", key="tf_nt_sample"):
        pick = vocab[int(np.random.choice(len(vocab), p=probs))]
        st.session_state.tf_seq = seq + [pick]
        st.rerun()
    if b3.button("🧹 Clear sequence", key="tf_nt_clear"):
        st.session_state.tf_seq = ["the"]
        st.rerun()

    st.caption(
        "Greedy always takes the top bar. Sampling rolls a weighted die — raise the "
        "temperature to flatten the bars (more random), lower it to sharpen them (more "
        "deterministic). This is the *exact* knob from the Temperature lab."
    )


def tab_transformer(ask_ai_widget: Callable[..., None]) -> None:
    """Render the Transformer tab. `ask_ai_widget` is injected by interview_prep.py."""
    st.subheader("🤖 Transformers, attention & next-token generation")
    st.caption("Free lab — pure math, no API keys needed.")

    big, sa, nt = st.tabs([
        "Big picture + math",
        "Self-attention (Q/K/V)",
        "Next-token generation",
    ])
    with big:
        _tf_big_picture()
    with sa:
        _tf_self_attention_lab()
    with nt:
        _tf_next_token_lab()

    st.divider()
    ask_ai_widget(
        "transformer",
        "Transformers, self-attention, Q/K/V, and next-token generation",
        "Transformer architecture: tokenize -> embed -> positional encoding -> attention "
        "blocks (Q=xWq, K=xWk, V=xWv; softmax(QK^T/sqrt(dk))V) -> unembedding to logits -> "
        "softmax -> sample next token -> loop. Self-attention: Q,K,V from same sequence. "
        "Cross-attention: Q from one sequence, K,V from another.",
    )
