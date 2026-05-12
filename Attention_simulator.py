import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Attention 메커니즘 시뮬레이터",
    page_icon="🧠",
    layout="wide",
)

# ─── Math ─────────────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

TOKENS = ["나는", "사과를", "매우", "좋아한다"]
TOKEN_COLORS = ["#3b82f6", "#a78bfa", "#34d399", "#f59e0b"]

BASE_VECS = np.array([
    [0.9, 0.1, 0.3, 0.5],
    [0.2, 0.8, 0.1, 0.7],
    [0.4, 0.3, 0.9, 0.2],
    [0.7, 0.6, 0.4, 0.8],
])
Wq = np.array([[0.6, -0.1], [0.2, 0.8], [-0.3, 0.5], [0.4, 0.3]])
Wk = np.array([[0.5, 0.2], [-0.1, 0.7], [0.3, -0.4], [0.6, 0.1]])
Wv = np.array([[0.4, 0.7], [0.3, -0.2], [0.8, 0.1], [-0.1, 0.6]])
dk = np.sqrt(2)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🧠 Attention 메커니즘 시뮬레이터")
st.caption("Scaled Dot-Product Attention · 단계별 인터랙티브 학습")
st.info("**Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V**")
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    query_idx = st.selectbox(
        "🔍 쿼리 토큰 선택",
        options=range(4),
        format_func=lambda i: f"★ {TOKENS[i]}",
        index=0,
    )

    st.divider()
    st.subheader("📐 가중치 행렬 커스텀")
    wq_00 = st.slider("Wq[0,0]", -1.0, 1.0, float(Wq[0, 0]), 0.05,
                      help="Query 행렬의 첫 번째 가중치")
    wk_00 = st.slider("Wk[0,0]", -1.0, 1.0, float(Wk[0, 0]), 0.05,
                      help="Key 행렬의 첫 번째 가중치")

    Wq_c = Wq.copy(); Wq_c[0, 0] = wq_00
    Wk_c = Wk.copy(); Wk_c[0, 0] = wk_00
    Q_c = BASE_VECS @ Wq_c
    K_c = BASE_VECS @ Wk_c
    V_c = BASE_VECS @ Wv

    q_vec   = Q_c[query_idx]
    scores  = (q_vec @ K_c.T) / dk
    weights = softmax(scores)
    output  = weights @ V_c

    st.divider()
    st.subheader("📊 Attention 가중치")
    weight_df = pd.DataFrame({"가중치": weights}, index=TOKENS)
    st.bar_chart(weight_df, color="#3b82f6", height=200)
    for tok, w in zip(TOKENS, weights):
        st.caption(f"{tok}: **{w:.4f}** ({w*100:.1f}%)")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Step 1: 임베딩",
    "🔢 Step 2: Q·K·V",
    "📐 Step 3: Score",
    "🌡️ Step 4: Softmax",
    "✨ Step 5: 출력 Z",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 임베딩
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("입력 임베딩 (dim=4)")
    st.caption("각 토큰을 4차원 벡터로 표현합니다. 실제 Transformer는 512~1024차원.")

    cols = st.columns(4)
    for i, (tok, col, color) in enumerate(zip(TOKENS, cols, TOKEN_COLORS)):
        with col:
            is_q = (i == query_idx)
            if is_q:
                st.success(f"**★ {tok}** ← 쿼리")
            else:
                st.info(f"**{tok}**")
            for d, v in enumerate(BASE_VECS[i]):
                st.metric(label=f"dim_{d}", value=f"{v:.2f}")

    st.divider()
    st.subheader("임베딩 행렬 히트맵")
    embed_df = pd.DataFrame(
        BASE_VECS,
        index=TOKENS,
        columns=[f"dim_{d}" for d in range(4)],
    )
    st.dataframe(
        embed_df.style
            .background_gradient(cmap="Blues", vmin=0, vmax=1)
            .format("{:.2f}"),
        use_container_width=True,
        height=212,
    )
    st.info(f"현재 쿼리 토큰: **{TOKENS[query_idx]}**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Q·K·V
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Q · K · V 투영 (4→2 차원)")
    st.caption("임베딩에 학습 가능한 행렬 Wq, Wk, Wv를 곱해 Q/K/V를 생성합니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔵 Wq (4×2)**")
        st.dataframe(
            pd.DataFrame(Wq_c, columns=["col0","col1"],
                         index=[f"row{i}" for i in range(4)])
              .style.background_gradient(cmap="Blues", vmin=-1, vmax=1).format("{:.2f}"),
            height=176,
        )
    with c2:
        st.markdown("**🟣 Wk (4×2)**")
        st.dataframe(
            pd.DataFrame(Wk_c, columns=["col0","col1"],
                         index=[f"row{i}" for i in range(4)])
              .style.background_gradient(cmap="Purples", vmin=-1, vmax=1).format("{:.2f}"),
            height=176,
        )
    with c3:
        st.markdown("**🟢 Wv (4×2)**")
        st.dataframe(
            pd.DataFrame(Wv, columns=["col0","col1"],
                         index=[f"row{i}" for i in range(4)])
              .style.background_gradient(cmap="Greens", vmin=-1, vmax=1).format("{:.2f}"),
            height=176,
        )

    st.divider()
    st.subheader("각 토큰의 Q / K / V 벡터")
    cols = st.columns(4)
    for i, (tok, col) in enumerate(zip(TOKENS, cols)):
        with col:
            is_q = (i == query_idx)
            label = f"{'★ ' if is_q else ''}{tok}"
            with st.expander(label, expanded=True):
                st.markdown(f"🔵 **Q** = `[{Q_c[i][0]:.3f}, {Q_c[i][1]:.3f}]`")
                st.markdown(f"🟣 **K** = `[{K_c[i][0]:.3f}, {K_c[i][1]:.3f}]`")
                st.markdown(f"🟢 **V** = `[{V_c[i][0]:.3f}, {V_c[i][1]:.3f}]`")

    with st.expander("💡 Q / K / V 개념 설명"):
        st.markdown("""
        - 🔵 **Q (Query)**: "내가 무엇을 찾고 있는가?" — 검색 쿼리
        - 🟣 **K (Key)**: "내가 어떤 정보를 갖고 있는가?" — 색인 키
        - 🟢 **V (Value)**: "실제로 전달할 정보는?" — 최종 값

        도서관 비유: Q는 검색어, K는 책 제목, V는 책 내용입니다.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Attention Score
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Attention Score = Q·Kᵀ / √dₖ")
    st.caption(f"쿼리 **{TOKENS[query_idx]}** 의 Q벡터와 모든 K벡터의 내적을 계산합니다.")

    cl, cr = st.columns([1, 1])
    with cl:
        st.markdown(f"#### Q = `[{q_vec[0]:.3f}, {q_vec[1]:.3f}]`")
        st.caption(f"√dₖ = {dk:.4f}")
        score_df = pd.DataFrame({
            "K 벡터": [f"[{K_c[i][0]:.3f}, {K_c[i][1]:.3f}]" for i in range(4)],
            "Q·K": [float(np.dot(q_vec, K_c[i])) for i in range(4)],
            "Score (÷√dₖ)": [float(scores[i]) for i in range(4)],
        }, index=TOKENS)
        st.dataframe(
            score_df.style
                .background_gradient(subset=["Score (÷√dₖ)"], cmap="Reds")
                .format({"Q·K": "{:.4f}", "Score (÷√dₖ)": "{:.4f}"}),
            use_container_width=True,
            height=212,
        )

    with cr:
        st.markdown("#### Score 시각화")
        score_chart_df = pd.DataFrame({"Score": scores}, index=TOKENS)
        st.bar_chart(score_chart_df, color="#f87171", height=260)

    best = int(np.argmax(scores))
    st.success(f"🔥 가장 높은 Score: **{TOKENS[best]}** ({scores[best]:.4f})")

    with st.expander("💡 왜 √dₖ 로 나누나요?"):
        st.markdown(f"""
        차원 dₖ가 커질수록 Q·K 내적값이 커져서 softmax 출력이 한 쪽으로 몰립니다 (기울기 소실).
        √dₖ = **{dk:.4f}** 로 나눠 스케일을 조정합니다.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Softmax
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Softmax → Attention 가중치")
    st.caption("Score를 합이 1인 확률 분포로 변환합니다.")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Score vs Softmax 비교")
        sc_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        compare_df = pd.DataFrame({
            "Score (정규화)": sc_norm,
            "Softmax Weight": weights,
        }, index=TOKENS)
        st.bar_chart(compare_df, height=280, color=["#f87171", "#34d399"])

    with cr:
        st.markdown("#### Attention 가중치 상세")
        weight_detail_df = pd.DataFrame({
            "가중치": weights,
            "퍼센트": [f"{w*100:.2f}%" for w in weights],
            "기여도": ["█" * int(w * 20) for w in weights],
        }, index=TOKENS)
        st.dataframe(
            weight_detail_df.style
                .background_gradient(subset=["가중치"], cmap="Greens")
                .format({"가중치": "{:.4f}"}),
            use_container_width=True,
            height=212,
        )
        st.metric("∑ weights", f"{weights.sum():.6f}", help="항상 1.0")

    st.divider()
    st.subheader("전체 Attention Weight 행렬")
    all_w = np.array([softmax((Q_c[i] @ K_c.T) / dk) for i in range(4)])
    heat_df = pd.DataFrame(
        all_w,
        index=[f"Query: {t}" for t in TOKENS],
        columns=[f"Key: {t}" for t in TOKENS],
    )
    st.dataframe(
        heat_df.style
            .background_gradient(cmap="YlOrRd", vmin=0, vmax=1)
            .format("{:.3f}"),
        use_container_width=True,
        height=212,
    )
    st.info(f"현재 쿼리 **{TOKENS[query_idx]}** 행을 주목하세요.")

    with st.expander("💡 Softmax 공식"):
        st.latex(r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: 출력 Z
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("출력 Z = Σ (attention_weight × V)")
    weighted_Vs = [weights[i] * V_c[i] for i in range(4)]

    st.markdown(f"#### 가중합 계산 과정 (쿼리: **{TOKENS[query_idx]}**)")
    cols = st.columns(4)
    for i, (tok, col) in enumerate(zip(TOKENS, cols)):
        wv = weighted_Vs[i]
        with col:
            st.metric(label=tok, value=f"w = {weights[i]:.3f}", delta=f"{weights[i]*100:.1f}%")
            st.caption(f"V = [{V_c[i][0]:.3f}, {V_c[i][1]:.3f}]")
            st.caption(f"→ [{wv[0]:.4f}, {wv[1]:.4f}]")

    st.divider()
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### 기여도 — Z[0] 차원")
        st.bar_chart(pd.DataFrame({"기여값": [weighted_Vs[i][0] for i in range(4)]}, index=TOKENS),
                     color="#3b82f6", height=220)
    with cr:
        st.markdown("#### 기여도 — Z[1] 차원")
        st.bar_chart(pd.DataFrame({"기여값": [weighted_Vs[i][1] for i in range(4)]}, index=TOKENS),
                     color="#a78bfa", height=220)

    st.divider()
    st.subheader(f"🎯 최종 출력 Z — \"{TOKENS[query_idx]}\"")
    best = int(np.argmax(weights))
    oc1, oc2, oc3 = st.columns(3)
    with oc1: st.metric("Z[0]", f"{output[0]:.4f}")
    with oc2: st.metric("Z[1]", f"{output[1]:.4f}")
    with oc3: st.metric("가장 주목한 토큰", TOKENS[best], f"{weights[best]*100:.1f}%")

    st.divider()
    st.subheader("📋 전체 계산 요약")
    summary_df = pd.DataFrame({
        "Score": scores,
        "Weight": weights,
        "Weight×V[0]": [weighted_Vs[i][0] for i in range(4)],
        "Weight×V[1]": [weighted_Vs[i][1] for i in range(4)],
    }, index=TOKENS)
    st.dataframe(
        summary_df.style
            .background_gradient(subset=["Weight"], cmap="Greens")
            .background_gradient(subset=["Score"], cmap="Reds")
            .format("{:.4f}"),
        use_container_width=True,
        height=212,
    )

    with st.expander("💡 최종 해석"):
        st.markdown(f"""
        **"{TOKENS[query_idx]}"** 의 출력 벡터 Z는 문장 내 모든 토큰의 V를
        attention weight로 **가중합산**한 결과입니다.

        | 항목 | 값 |
        |---|---|
        | 쿼리 토큰 | **{TOKENS[query_idx]}** |
        | 출력 Z | `[{output[0]:.4f}, {output[1]:.4f}]` |
        | 가장 주목한 토큰 | **{TOKENS[best]}** ({weights[best]*100:.1f}%) |
        """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Scaled Dot-Product Attention · Vaswani et al. (2017) 'Attention is All You Need'")