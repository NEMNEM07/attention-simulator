import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attention 메커니즘 시뮬레이터",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #070b12;
    color: #d0ddf0;
}
.stApp { background-color: #070b12; }

/* Metric cards */
.metric-card {
    background: #0d1420;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.step-header {
    font-size: 11px;
    letter-spacing: 3px;
    margin-bottom: 10px;
    font-weight: 700;
}
.formula-box {
    background: #0a1020;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 15px;
    text-align: center;
    color: #fbbf24;
    font-weight: 600;
    margin: 10px 0;
}
.token-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 3px;
    font-weight: 600;
}
.info-box {
    background: #0a1825;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 12px;
    color: #7098c8;
    margin-top: 8px;
    line-height: 1.7;
}
h1 { color: #e8f0ff !important; }
h2, h3 { color: #93c5fd !important; }
.stSlider label { color: #7098c8 !important; font-size: 12px !important; }
div[data-testid="stSelectbox"] label { color: #7098c8 !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Math helpers ─────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def matmul_vec(M, v):
    return M @ v

# ─── Toy data ─────────────────────────────────────────────────────────────────
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

def compute_qkv(embeddings):
    Q = embeddings @ Wq
    K = embeddings @ Wk
    V = embeddings @ Wv
    return Q, K, V

def compute_attention(Q, K, V, query_idx):
    q = Q[query_idx]
    scores = (q @ K.T) / dk
    weights = softmax(scores)
    output = weights @ V
    return scores, weights, output

# ─── Compute ──────────────────────────────────────────────────────────────────
Q, K, V = compute_qkv(BASE_VECS)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 20px 0;'>
  <div style='font-size:10px; letter-spacing:4px; color:#3b82f6; margin-bottom:6px;'>
    ATTENTION IS ALL YOU NEED
  </div>
  <h1 style='font-size:26px; margin:0; letter-spacing:-0.5px;'>
    🧠 Attention 메커니즘 시뮬레이터
  </h1>
  <div style='font-size:11px; color:#445566; margin-top:6px;'>
    Scaled Dot-Product Attention · 단계별 인터랙티브 학습
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="formula-box">Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V</div>',
            unsafe_allow_html=True)

st.divider()

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    st.markdown("---")
    
    query_idx = st.selectbox(
        "🔍 쿼리 토큰 선택",
        options=range(4),
        format_func=lambda i: f"★ {TOKENS[i]}",
        index=0,
    )
    
    st.markdown("---")
    st.markdown("### 📐 가중치 행렬 커스텀")
    st.caption("Wq[0,0] (Query 첫 번째 가중치)")
    wq_00 = st.slider("Wq[0,0]", -1.0, 1.0, float(Wq[0, 0]), 0.05)
    st.caption("Wk[0,0] (Key 첫 번째 가중치)")
    wk_00 = st.slider("Wk[0,0]", -1.0, 1.0, float(Wk[0, 0]), 0.05)
    
    # Apply custom weights
    Wq_custom = Wq.copy()
    Wk_custom = Wk.copy()
    Wq_custom[0, 0] = wq_00
    Wk_custom[0, 0] = wk_00
    
    Q_c = BASE_VECS @ Wq_custom
    K_c = BASE_VECS @ Wk_custom
    V_c = BASE_VECS @ Wv
    
    scores, weights, output = compute_attention(Q_c, K_c, V_c, query_idx)
    
    st.markdown("---")
    st.markdown("### 📊 현재 Attention 가중치")
    for i, (tok, w, c) in enumerate(zip(TOKENS, weights, TOKEN_COLORS)):
        pct = w * 100
        st.markdown(f"""
        <div style='margin-bottom:6px;'>
          <span style='color:{c}; font-size:12px; font-weight:700;'>{tok}</span>
          <div style='background:#0d1420; border-radius:4px; height:16px; margin-top:3px; overflow:hidden;'>
            <div style='background:{c}; height:100%; width:{pct:.1f}%; border-radius:4px; 
                        display:flex; align-items:center; padding-left:6px;'>
            </div>
          </div>
          <span style='font-size:10px; color:#556;'>{w:.4f} ({pct:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Main Content: Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Step 1: 임베딩",
    "🔢 Step 2: Q·K·V",
    "📐 Step 3: Score",
    "🌡️ Step 4: Softmax",
    "✨ Step 5: 출력 Z",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1: 입력 임베딩
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="step-header" style="color:#f59e0b;">STEP 1 · 입력 임베딩 (dim=4)</div>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (tok, col, color) in enumerate(zip(TOKENS, cols, TOKEN_COLORS)):
        with col:
            is_query = (i == query_idx)
            border = f"2px solid {color}" if is_query else "1px solid #1a2540"
            star = "★ " if is_query else ""
            st.markdown(f"""
            <div class='metric-card' style='border:{border}; text-align:center;'>
              <div style='color:{color}; font-size:14px; font-weight:700; margin-bottom:8px;'>
                {star}{tok}
              </div>
              {"<div style='font-size:10px; color:#3b82f6; margin-bottom:6px;'>← 쿼리 토큰</div>" if is_query else ""}
            </div>
            """, unsafe_allow_html=True)

    # Heatmap of embeddings
    fig = go.Figure(data=go.Heatmap(
        z=BASE_VECS,
        x=[f"dim_{d}" for d in range(4)],
        y=TOKENS,
        colorscale=[[0, "#070b12"], [0.5, "#1a3060"], [1, "#3b82f6"]],
        text=[[f"{v:.2f}" for v in row] for row in BASE_VECS],
        texttemplate="%{text}",
        textfont={"size": 13, "family": "IBM Plex Mono"},
        showscale=True,
        colorbar=dict(bgcolor="#0d1420", tickfont=dict(color="#d0ddf0")),
    ))
    fig.update_layout(
        title=dict(text="입력 임베딩 행렬 (4×4)", font=dict(color="#d0ddf0", size=14)),
        paper_bgcolor="#070b12", plot_bgcolor="#0d1420",
        font=dict(color="#d0ddf0", family="IBM Plex Mono"),
        height=280, margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#1a2540"),
        yaxis=dict(gridcolor="#1a2540"),
    )
    # Highlight query row
    fig.add_shape(type="rect",
        x0=-0.5, x1=3.5,
        y0=query_idx - 0.5, y1=query_idx + 0.5,
        line=dict(color=TOKEN_COLORS[query_idx], width=2.5),
        fillcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class='info-box'>
    📖 <b>개념</b>: 각 토큰을 고정 길이의 벡터로 표현합니다. 실제 Transformer에서는 512~1024차원이지만,
    여기서는 4차원으로 단순화했습니다.<br>
    현재 쿼리 토큰: <b style='color:{TOKEN_COLORS[query_idx]};'>{TOKENS[query_idx]}</b>
    (강조된 행)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2: Q·K·V 계산
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="step-header" style="color:#a78bfa;">STEP 2 · Q · K · V 투영 (4→2 차원)</div>',
                unsafe_allow_html=True)

    col_wq, col_wk, col_wv = st.columns(3)
    
    def matrix_heatmap(M, title, color_high):
        fig = go.Figure(data=go.Heatmap(
            z=M,
            colorscale=[[0, "#070b12"], [0.5, "#0d1420"], [1, color_high]],
            text=[[f"{v:.1f}" for v in row] for row in M],
            texttemplate="%{text}",
            textfont={"size": 11, "family": "IBM Plex Mono"},
            showscale=False,
            zmid=0,
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#d0ddf0", size=12)),
            paper_bgcolor="#070b12", plot_bgcolor="#0d1420",
            font=dict(color="#d0ddf0", family="IBM Plex Mono"),
            height=180, margin=dict(l=30, r=10, t=40, b=10),
        )
        return fig

    with col_wq:
        st.plotly_chart(matrix_heatmap(Wq_custom, "Wq (4×2)", "#3b82f6"),
                       use_container_width=True)
    with col_wk:
        st.plotly_chart(matrix_heatmap(Wk_custom, "Wk (4×2)", "#a78bfa"),
                       use_container_width=True)
    with col_wv:
        st.plotly_chart(matrix_heatmap(Wv, "Wv (4×2)", "#34d399"),
                       use_container_width=True)

    st.markdown("#### 각 토큰의 Q / K / V 벡터")
    cols = st.columns(4)
    for i, (tok, col, color) in enumerate(zip(TOKENS, cols, TOKEN_COLORS)):
        with col:
            q_vec = Q_c[i]
            k_vec = K_c[i]
            v_vec = V_c[i]
            st.markdown(f"""
            <div class='metric-card' style='border: 1px solid {color}44;'>
              <div style='color:{color}; font-size:13px; font-weight:700; margin-bottom:8px;'>{tok}</div>
              <div style='font-size:10px; color:#3b82f6; margin-bottom:3px;'>Q</div>
              <div style='font-size:11px; color:#93c5fd; margin-bottom:6px;'>
                [{q_vec[0]:.3f}, {q_vec[1]:.3f}]
              </div>
              <div style='font-size:10px; color:#a78bfa; margin-bottom:3px;'>K</div>
              <div style='font-size:11px; color:#c4b5fd; margin-bottom:6px;'>
                [{k_vec[0]:.3f}, {k_vec[1]:.3f}]
              </div>
              <div style='font-size:10px; color:#34d399; margin-bottom:3px;'>V</div>
              <div style='font-size:11px; color:#6ee7b7;'>
                [{v_vec[0]:.3f}, {v_vec[1]:.3f}]
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-box'>
    📖 <b>개념</b>: 임베딩 벡터에 학습 가능한 가중치 행렬 Wq, Wk, Wv를 곱해 Q(Query), K(Key), V(Value)를 생성합니다.<br>
    • <b style='color:#3b82f6;'>Q</b>: "내가 무엇을 찾고 있는가?" (검색 쿼리)<br>
    • <b style='color:#a78bfa;'>K</b>: "내가 어떤 정보를 갖고 있는가?" (색인 키)<br>
    • <b style='color:#34d399;'>V</b>: "실제로 전달할 정보는?" (값)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3: Attention Score
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="step-header" style="color:#f87171;">STEP 3 · Attention Score = Q·Kᵀ / √dₖ</div>',
                unsafe_allow_html=True)

    q_vec = Q_c[query_idx]
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown(f"#### 쿼리 Q = [{q_vec[0]:.3f}, {q_vec[1]:.3f}]")
        
        score_data = []
        for i, tok in enumerate(TOKENS):
            k_vec = K_c[i]
            raw_dot = np.dot(q_vec, k_vec)
            scaled = raw_dot / dk
            score_data.append({
                "토큰": tok,
                "K 벡터": f"[{k_vec[0]:.3f}, {k_vec[1]:.3f}]",
                "Q·K": f"{raw_dot:.4f}",
                "÷√dₖ": f"{scaled:.4f}",
            })
        
        for i, row in enumerate(score_data):
            is_max = scores[i] == scores.max()
            border = "1px solid #f8717188" if is_max else "1px solid #1a2540"
            st.markdown(f"""
            <div class='metric-card' style='border:{border}; margin-bottom:6px;'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='color:{TOKEN_COLORS[i]}; font-weight:700; font-size:13px;'>
                  {row["토큰"]} {"🔥" if is_max else ""}
                </span>
                <span style='font-size:14px; color:{"#f87171" if is_max else "#d0ddf0"}; font-weight:700;'>
                  {row["÷√dₖ"]}
                </span>
              </div>
              <div style='font-size:10px; color:#556; margin-top:4px;'>
                K = {row["K 벡터"]} &nbsp;|&nbsp; Q·K = {row["Q·K"]}
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # Bar chart of scores
        fig = go.Figure(data=go.Bar(
            x=TOKENS,
            y=scores,
            marker=dict(
                color=scores,
                colorscale=[[0, "#1a2540"], [0.5, "#7f1d1d"], [1, "#f87171"]],
                line=dict(color="#f8717188", width=1),
            ),
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
            textfont=dict(color="#f87171", size=11),
        ))
        fig.update_layout(
            title=dict(text="Raw Attention Scores", font=dict(color="#d0ddf0", size=13)),
            paper_bgcolor="#070b12", plot_bgcolor="#0d1420",
            font=dict(color="#d0ddf0", family="IBM Plex Mono"),
            height=280, margin=dict(l=20, r=20, t=50, b=40),
            xaxis=dict(gridcolor="#1a2540"),
            yaxis=dict(gridcolor="#1a2540", zeroline=True, zerolinecolor="#334"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class='info-box'>
    📖 <b>개념</b>: Q와 각 K의 내적(dot product)으로 "유사도"를 계산합니다.
    √dₖ로 나누는 이유는 차원이 커질수록 내적값이 커져 softmax의 기울기가 소실되는 것을 방지하기 위해서입니다.<br>
    현재 dₖ = {dk:.4f}, √dₖ = {dk:.4f}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4: Softmax → Weights
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="step-header" style="color:#34d399;">STEP 4 · Softmax → Attention 가중치</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Before/After softmax comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Raw Score",
            x=TOKENS, y=scores,
            marker_color="#f87171aa",
            offsetgroup=0,
        ))
        fig.add_trace(go.Bar(
            name="Softmax Weight",
            x=TOKENS, y=weights,
            marker_color="#34d399",
            offsetgroup=1,
            text=[f"{w:.3f}" for w in weights],
            textposition="outside",
            textfont=dict(color="#34d399", size=11),
        ))
        fig.update_layout(
            title=dict(text="Softmax 전후 비교", font=dict(color="#d0ddf0", size=13)),
            paper_bgcolor="#070b12", plot_bgcolor="#0d1420",
            font=dict(color="#d0ddf0", family="IBM Plex Mono"),
            height=300, margin=dict(l=20, r=20, t=50, b=40),
            barmode="group",
            xaxis=dict(gridcolor="#1a2540"),
            yaxis=dict(gridcolor="#1a2540"),
            legend=dict(bgcolor="#0d1420", bordercolor="#1a2540", borderwidth=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Pie chart
        fig_pie = go.Figure(data=go.Pie(
            labels=TOKENS,
            values=weights,
            hole=0.5,
            marker=dict(colors=TOKEN_COLORS, line=dict(color="#070b12", width=2)),
            textfont=dict(family="IBM Plex Mono", size=12),
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            title=dict(text=f'"{TOKENS[query_idx]}"의 Attention 분포', font=dict(color="#d0ddf0", size=13)),
            paper_bgcolor="#070b12",
            font=dict(color="#d0ddf0", family="IBM Plex Mono"),
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(bgcolor="#0d1420", bordercolor="#1a2540"),
            annotations=[dict(text=TOKENS[query_idx], x=0.5, y=0.5, font_size=16,
                             font_color=TOKEN_COLORS[query_idx], showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Attention heatmap (전체 토큰 × 토큰)
    st.markdown("#### 전체 Attention Weight 행렬 (모든 쿼리 토큰)")
    all_weights = np.array([softmax((Q_c[i] @ K_c.T) / dk) for i in range(4)])
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=all_weights,
        x=[f"Key: {t}" for t in TOKENS],
        y=[f"Query: {t}" for t in TOKENS],
        colorscale=[[0, "#070b12"], [0.3, "#1a3060"], [0.7, "#f87171"], [1, "#fbbf24"]],
        text=[[f"{v:.3f}" for v in row] for row in all_weights],
        texttemplate="%{text}",
        textfont={"size": 12, "family": "IBM Plex Mono"},
        showscale=True,
        colorbar=dict(bgcolor="#0d1420", tickfont=dict(color="#d0ddf0")),
    ))
    fig_heat.update_layout(
        paper_bgcolor="#070b12", plot_bgcolor="#0d1420",
        font=dict(color="#d0ddf0", family="IBM Plex Mono"),
        height=260, margin=dict(l=100, r=20, t=20, b=80),
        xaxis=dict(gridcolor="#1a2540"),
        yaxis=dict(gridcolor="#1a2540"),
    )
    # Highlight query row
    fig_heat.add_shape(type="rect",
        x0=-0.5, x1=3.5,
        y0=query_idx - 0.5, y1=query_idx + 0.5,
        line=dict(color=TOKEN_COLORS[query_idx], width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(f"""
    <div class='info-box'>
    📖 <b>개념</b>: softmax는 모든 score를 합이 1인 확률 분포로 변환합니다.
    값이 클수록 해당 토큰에 더 많이 "주목(attend)"합니다.<br>
    ∑ weights = {weights.sum():.6f} (항상 1.0)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5: 출력 Z
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="step-header" style="color:#fbbf24;">STEP 5 · 출력 Z = Σ (attention_weight × V)</div>',
                unsafe_allow_html=True)

    # Show weighted V vectors
    st.markdown(f"#### 가중합 계산 과정 (쿼리: {TOKENS[query_idx]})")
    
    weighted_Vs = [weights[i] * V_c[i] for i in range(4)]
    
    cols = st.columns(4)
    for i, (tok, col, color) in enumerate(zip(TOKENS, cols, TOKEN_COLORS)):
        with col:
            w = weights[i]
            wv = weighted_Vs[i]
            st.markdown(f"""
            <div class='metric-card' style='border:1px solid {color}44; text-align:center;'>
              <div style='color:{color}; font-size:13px; font-weight:700;'>{tok}</div>
              <div style='font-size:22px; color:#34d399; font-weight:700; margin:6px 0;'>
                {w:.3f}
              </div>
              <div style='font-size:10px; color:#556; margin-bottom:4px;'>× V</div>
              <div style='font-size:11px; color:#6ee7b7;'>
                [{V_c[i][0]:.3f}, {V_c[i][1]:.3f}]
              </div>
              <div style='font-size:10px; color:#888; margin-top:4px;'>= [{wv[0]:.4f}, {wv[1]:.4f}]</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### ➕ 합산 → 출력 Z")
    
    # Stacked bar showing contribution of each token to output
    fig_stack = go.Figure()
    for i, (tok, color) in enumerate(zip(TOKENS, TOKEN_COLORS)):
        fig_stack.add_trace(go.Bar(
            name=tok,
            x=["Z[0]", "Z[1]"],
            y=[weighted_Vs[i][0], weighted_Vs[i][1]],
            marker_color=color,
            text=[f"{weighted_Vs[i][0]:.4f}", f"{weighted_Vs[i][1]:.4f}"],
            textposition="inside",
            textfont=dict(size=10),
        ))
    
    fig_stack.update_layout(
        title=dict(text=f"각 토큰의 기여도 (출력 Z)", font=dict(color="#d0ddf0", size=13)),
        barmode="stack",
        paper_bgcolor="#070b12", plot_bgcolor="#0d1420",
        font=dict(color="#d0ddf0", family="IBM Plex Mono"),
        height=280, margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#1a2540"),
        yaxis=dict(gridcolor="#1a2540"),
        legend=dict(bgcolor="#0d1420", bordercolor="#1a2540", borderwidth=1),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # Final output display
    st.markdown(f"""
    <div style='background:#091830; border:1px solid #fbbf2444; border-radius:10px; padding:16px 20px; text-align:center;'>
      <div style='font-size:12px; color:#fbbf24; letter-spacing:2px; margin-bottom:10px;'>
        최종 출력 Z ("{TOKENS[query_idx]}")
      </div>
      <div style='display:flex; justify-content:center; gap:12px;'>
        <div style='background:#1a3060; border:1px solid #fbbf2444; border-radius:8px; 
                    padding:10px 20px; font-size:20px; font-weight:700; color:#fde68a;'>
          Z[0] = {output[0]:.4f}
        </div>
        <div style='background:#1a3060; border:1px solid #fbbf2444; border-radius:8px; 
                    padding:10px 20px; font-size:20px; font-weight:700; color:#fde68a;'>
          Z[1] = {output[1]:.4f}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-box' style='margin-top:12px;'>
    📖 <b>최종 해석</b>: "{TOKENS[query_idx]}"의 출력 벡터 Z는 문장 내 모든 토큰의 V를 
    attention weight로 가중합산한 결과입니다. 이 벡터는 해당 토큰이 
    <b style='color:{TOKEN_COLORS[query_idx]};'>다른 토큰들과의 관계를 반영한 새로운 표현</b>입니다.<br>
    가장 많이 주목한 토큰: <b style='color:{TOKEN_COLORS[int(np.argmax(weights))]};'>
    {TOKENS[int(np.argmax(weights))]} ({weights.max()*100:.1f}%)</b>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; font-size:10px; color:#334; padding:10px 0;'>
  Scaled Dot-Product Attention · Vaswani et al. (2017) "Attention is All You Need"<br>
  dim=4 임베딩, dim_k=2 투영, 4개 토큰 한국어 예제
</div>
""", unsafe_allow_html=True)