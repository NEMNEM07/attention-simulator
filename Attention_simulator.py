import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import io

st.set_page_config(
    page_title="Attention 메커니즘 시뮬레이터",
    page_icon="🧠",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
.stApp { background-color: #070b12; color: #d0ddf0; }
.metric-card {
    background: #0d1420; border: 1px solid #1a2540;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.step-header {
    font-size: 11px; letter-spacing: 3px; margin-bottom: 10px; font-weight: 700;
}
.formula-box {
    background: #0a1020; border: 1px solid #1a2540; border-radius: 8px;
    padding: 12px 16px; font-size: 15px; text-align: center;
    color: #fbbf24; font-weight: 600; margin: 10px 0;
}
.info-box {
    background: #0a1825; border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    font-size: 12px; color: #7098c8; margin-top: 8px; line-height: 1.7;
}
h1, h2, h3 { color: #e8f0ff !important; }
</style>
""", unsafe_allow_html=True)

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

# ─── Matplotlib theme helper ──────────────────────────────────────────────────
BG  = "#070b12"
BG2 = "#0d1420"
FG  = "#d0ddf0"
GRID = "#1a2540"

def dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    return fig, ax

def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:10px 0 20px 0;'>
  <div style='font-size:10px; letter-spacing:4px; color:#3b82f6; margin-bottom:6px;'>
    ATTENTION IS ALL YOU NEED
  </div>
  <h1 style='font-size:26px; margin:0;'>🧠 Attention 메커니즘 시뮬레이터</h1>
  <div style='font-size:11px; color:#445566; margin-top:6px;'>
    Scaled Dot-Product Attention · 단계별 인터랙티브 학습
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="formula-box">Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V</div>',
            unsafe_allow_html=True)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
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
    wq_00 = st.slider("Wq[0,0]", -1.0, 1.0, float(Wq[0, 0]), 0.05)
    wk_00 = st.slider("Wk[0,0]", -1.0, 1.0, float(Wk[0, 0]), 0.05)

    Wq_c = Wq.copy(); Wq_c[0, 0] = wq_00
    Wk_c = Wk.copy(); Wk_c[0, 0] = wk_00
    Q_c = BASE_VECS @ Wq_c
    K_c = BASE_VECS @ Wk_c
    V_c = BASE_VECS @ Wv

    q_vec   = Q_c[query_idx]
    scores  = (q_vec @ K_c.T) / dk
    weights = softmax(scores)
    output  = weights @ V_c

    st.markdown("---")
    st.markdown("### 📊 현재 Attention 가중치")
    for i, (tok, w, c) in enumerate(zip(TOKENS, weights, TOKEN_COLORS)):
        pct = w * 100
        st.markdown(f"""
        <div style='margin-bottom:6px;'>
          <span style='color:{c}; font-size:12px; font-weight:700;'>{tok}</span>
          <div style='background:#0d1420; border-radius:4px; height:14px; margin-top:3px; overflow:hidden;'>
            <div style='background:{c}; height:100%; width:{pct:.1f}%; border-radius:4px;'></div>
          </div>
          <span style='font-size:10px; color:#556;'>{w:.4f} ({pct:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Step 1: 임베딩",
    "🔢 Step 2: Q·K·V",
    "📐 Step 3: Score",
    "🌡️ Step 4: Softmax",
    "✨ Step 5: 출력 Z",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 임베딩 히트맵
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="step-header" style="color:#f59e0b;">STEP 1 · 입력 임베딩 (dim=4)</div>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (tok, col, color) in enumerate(zip(TOKENS, cols, TOKEN_COLORS)):
        with col:
            is_q  = (i == query_idx)
            border = f"2px solid {color}" if is_q else "1px solid #1a2540"
            st.markdown(f"""
            <div class='metric-card' style='border:{border}; text-align:center;'>
              <div style='color:{color}; font-size:14px; font-weight:700;'>
                {"★ " if is_q else ""}{tok}
              </div>
              {"<div style='font-size:10px;color:#3b82f6;margin-top:4px;'>← 쿼리 토큰</div>" if is_q else ""}
            </div>
            """, unsafe_allow_html=True)

    # Heatmap via matplotlib
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    im = ax.imshow(BASE_VECS, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"dim_{d}" for d in range(4)], color=FG, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(TOKENS, color=FG, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{BASE_VECS[i,j]:.2f}", ha="center", va="center",
                    color="white" if BASE_VECS[i,j] > 0.5 else "#93c5fd", fontsize=9)
    # Highlight query row
    rect = mpatches.FancyBboxPatch((-0.5, query_idx - 0.5), 4, 1,
        boxstyle="round,pad=0.05", linewidth=2,
        edgecolor=TOKEN_COLORS[query_idx], facecolor="none")
    ax.add_patch(rect)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=FG, labelsize=8)
    ax.set_title("입력 임베딩 행렬 (4×4)", color=FG, fontsize=11, pad=10)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID)
    fig_to_st(fig)

    st.markdown(f"""
    <div class='info-box'>
    📖 <b>개념</b>: 각 토큰을 고정 길이 벡터로 표현합니다. 실제 Transformer에서는 512~1024차원이지만
    여기서는 4차원으로 단순화했습니다.<br>
    현재 쿼리 토큰: <b style='color:{TOKEN_COLORS[query_idx]};'>{TOKENS[query_idx]}</b> (강조된 행)
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Q·K·V
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="step-header" style="color:#a78bfa;">STEP 2 · Q · K · V 투영 (4→2 차원)</div>',
                unsafe_allow_html=True)

    # Show W matrices
    def plot_matrix(M, title, cmap):
        fig, ax = plt.subplots(figsize=(2.5, 2.2))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["col0", "col1"], color=FG, fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels([f"r{i}" for i in range(4)], color=FG, fontsize=8)
        for i in range(4):
            for j in range(2):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", color=FG, fontsize=8)
        ax.set_title(title, color=FG, fontsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=BG, dpi=120)
        buf.seek(0); plt.close(fig)
        return buf

    c1, c2, c3 = st.columns(3)
    with c1: st.image(plot_matrix(Wq_c, "Wq (4×2)", "Blues"), use_column_width=True)
    with c2: st.image(plot_matrix(Wk_c, "Wk (4×2)", "Purples"), use_column_width=True)
    with c3: st.image(plot_matrix(Wv,   "Wv (4×2)", "Greens"), use_column_width=True)

    st.markdown("#### 각 토큰의 Q / K / V 벡터")
    cols = st.columns(4)
    for i, (tok, col, color) in enumerate(zip(TOKENS, cols, TOKEN_COLORS)):
        q_v = Q_c[i]; k_v = K_c[i]; v_v = V_c[i]
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border:1px solid {color}44;'>
              <div style='color:{color};font-size:13px;font-weight:700;margin-bottom:8px;'>{tok}</div>
              <div style='font-size:10px;color:#3b82f6;'>Q</div>
              <div style='font-size:11px;color:#93c5fd;margin-bottom:5px;'>[{q_v[0]:.3f}, {q_v[1]:.3f}]</div>
              <div style='font-size:10px;color:#a78bfa;'>K</div>
              <div style='font-size:11px;color:#c4b5fd;margin-bottom:5px;'>[{k_v[0]:.3f}, {k_v[1]:.3f}]</div>
              <div style='font-size:10px;color:#34d399;'>V</div>
              <div style='font-size:11px;color:#6ee7b7;'>[{v_v[0]:.3f}, {v_v[1]:.3f}]</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    📖 <b>개념</b>: 임베딩에 학습 가능한 행렬 Wq, Wk, Wv를 곱해 Q/K/V를 생성합니다.<br>
    • <b style='color:#3b82f6;'>Q</b>: "내가 무엇을 찾고 있는가?" (검색 쿼리)<br>
    • <b style='color:#a78bfa;'>K</b>: "내가 어떤 정보를 갖고 있는가?" (색인 키)<br>
    • <b style='color:#34d399;'>V</b>: "실제로 전달할 정보는?" (값)
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Attention Score
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="step-header" style="color:#f87171;">STEP 3 · Attention Score = Q·Kᵀ / √dₖ</div>',
                unsafe_allow_html=True)

    cl, cr = st.columns(2)

    with cl:
        st.markdown(f"#### 쿼리 Q = [{q_vec[0]:.3f}, {q_vec[1]:.3f}]")
        for i, tok in enumerate(TOKENS):
            raw_dot = float(np.dot(q_vec, K_c[i]))
            scaled  = raw_dot / dk
            is_max  = (i == int(np.argmax(scores)))
            border  = "1px solid #f8717188" if is_max else "1px solid #1a2540"
            st.markdown(f"""
            <div class='metric-card' style='border:{border};margin-bottom:5px;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='color:{TOKEN_COLORS[i]};font-weight:700;font-size:13px;'>
                  {tok} {"🔥" if is_max else ""}
                </span>
                <span style='font-size:15px;color:{"#f87171" if is_max else FG};font-weight:700;'>
                  {scaled:.4f}
                </span>
              </div>
              <div style='font-size:10px;color:#556;margin-top:3px;'>
                K=[{K_c[i][0]:.3f},{K_c[i][1]:.3f}] &nbsp;|&nbsp; Q·K={raw_dot:.4f}
              </div>
            </div>
            """, unsafe_allow_html=True)

    with cr:
        fig, ax = dark_fig(5, 4)
        colors  = [TOKEN_COLORS[i] for i in range(4)]
        bars = ax.bar(TOKENS, scores, color=colors, edgecolor=GRID, linewidth=0.8)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{s:.4f}", ha="center", va="bottom", color="#f87171", fontsize=9)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_title("Raw Attention Scores", color=FG, fontsize=11)
        ax.set_ylabel("Score", color=FG, fontsize=9)
        ax.tick_params(axis="x", colors=FG); ax.tick_params(axis="y", colors=FG)
        fig_to_st(fig)

    st.markdown(f"""
    <div class='info-box'>
    📖 <b>개념</b>: Q와 각 K의 내적으로 유사도를 계산하고 √dₖ({dk:.3f})로 나눠 스케일링합니다.
    스케일링이 없으면 내적값이 커져 softmax 기울기 소실이 발생합니다.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Softmax
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="step-header" style="color:#34d399;">STEP 4 · Softmax → Attention 가중치</div>',
                unsafe_allow_html=True)

    cl, cr = st.columns(2)

    with cl:
        # Grouped bar: raw score vs softmax weight
        fig, ax = dark_fig(5, 4)
        x = np.arange(4)
        w = 0.35
        # Normalize scores to same scale for visual comparison
        sc_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        ax.bar(x - w/2, sc_norm, w, label="Score (정규화)", color="#f87171", alpha=0.8, edgecolor=GRID)
        ax.bar(x + w/2, weights, w, label="Softmax Weight", color="#34d399", alpha=0.9, edgecolor=GRID)
        for xi, wt in zip(x + w/2, weights):
            ax.text(xi, wt + 0.01, f"{wt:.3f}", ha="center", va="bottom", color="#34d399", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(TOKENS, color=FG, fontsize=10)
        ax.set_title("Softmax 전후 비교", color=FG, fontsize=11)
        ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=FG, fontsize=8)
        fig_to_st(fig)

    with cr:
        # Pie chart
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        wedges, texts, autotexts = ax.pie(
            weights, labels=TOKENS, colors=TOKEN_COLORS,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor=BG, linewidth=2),
            textprops=dict(color=FG, fontsize=10),
        )
        for at in autotexts: at.set_color(BG); at.set_fontsize(9)
        ax.set_title(f'"{TOKENS[query_idx]}"의 Attention 분포', color=FG, fontsize=11)
        fig_to_st(fig)

    # Full attention weight heatmap
    st.markdown("#### 전체 Attention Weight 행렬")
    all_w = np.array([softmax((Q_c[i] @ K_c.T) / dk) for i in range(4)])
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
    im = ax.imshow(all_w, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"Key: {t}" for t in TOKENS], color=FG, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"Query: {t}" for t in TOKENS], color=FG, fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{all_w[i,j]:.3f}", ha="center", va="center",
                    color="black" if all_w[i,j] > 0.5 else FG, fontsize=9)
    rect = mpatches.FancyBboxPatch((-0.5, query_idx-0.5), 4, 1,
        boxstyle="round,pad=0.05", linewidth=2.5,
        edgecolor=TOKEN_COLORS[query_idx], facecolor="none")
    ax.add_patch(rect)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=FG, labelsize=8)
    ax.set_title("Attention Weight Matrix (모든 쿼리)", color=FG, fontsize=11, pad=10)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID)
    fig_to_st(fig)

    st.markdown(f"""
    <div class='info-box'>
    📖 <b>개념</b>: softmax는 score를 합이 1인 확률 분포로 변환합니다.
    값이 클수록 해당 토큰에 더 많이 "주목(attend)"합니다.<br>
    ∑ weights = {weights.sum():.6f} (항상 1.0)
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: 출력 Z
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="step-header" style="color:#fbbf24;">STEP 5 · 출력 Z = Σ (attention_weight × V)</div>',
                unsafe_allow_html=True)

    weighted_Vs = [weights[i] * V_c[i] for i in range(4)]

    st.markdown(f"#### 가중합 계산 과정 (쿼리: {TOKENS[query_idx]})")
    cols = st.columns(4)
    for i, (tok, col, color) in enumerate(zip(TOKENS, cols, TOKEN_COLORS)):
        wv = weighted_Vs[i]
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border:1px solid {color}44;text-align:center;'>
              <div style='color:{color};font-size:13px;font-weight:700;'>{tok}</div>
              <div style='font-size:22px;color:#34d399;font-weight:700;margin:6px 0;'>
                {weights[i]:.3f}
              </div>
              <div style='font-size:10px;color:#556;margin-bottom:4px;'>× V</div>
              <div style='font-size:11px;color:#6ee7b7;'>[{V_c[i][0]:.3f}, {V_c[i][1]:.3f}]</div>
              <div style='font-size:10px;color:#888;margin-top:4px;'>= [{wv[0]:.4f}, {wv[1]:.4f}]</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### ➕ 합산 → 출력 Z")

    # Stacked bar
    fig, ax = dark_fig(5, 3.5)
    x     = np.arange(2)
    bottoms = [0.0, 0.0]
    for i, (tok, color) in enumerate(zip(TOKENS, TOKEN_COLORS)):
        vals = [weighted_Vs[i][0], weighted_Vs[i][1]]
        ax.bar(x, vals, bottom=bottoms, color=color, label=tok,
               edgecolor=BG, linewidth=0.8, alpha=0.9)
        for xi, v, b in zip(x, vals, bottoms):
            if abs(v) > 0.01:
                ax.text(xi, b + v/2, f"{v:.3f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        bottoms = [bottoms[j] + weighted_Vs[i][j] for j in range(2)]
    ax.set_xticks(x); ax.set_xticklabels(["Z[0]", "Z[1]"], color=FG, fontsize=12)
    ax.set_title(f"각 토큰의 기여도 → 출력 Z", color=FG, fontsize=11)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="upper right")
    fig_to_st(fig)

    # Final output
    st.markdown(f"""
    <div style='background:#091830;border:1px solid #fbbf2444;border-radius:10px;
                padding:16px 20px;text-align:center;margin-top:10px;'>
      <div style='font-size:12px;color:#fbbf24;letter-spacing:2px;margin-bottom:12px;'>
        최종 출력 Z ("{TOKENS[query_idx]}")
      </div>
      <div style='display:flex;justify-content:center;gap:16px;'>
        <div style='background:#1a3060;border:1px solid #fbbf2444;border-radius:8px;
                    padding:12px 24px;font-size:20px;font-weight:700;color:#fde68a;'>
          Z[0] = {output[0]:.4f}
        </div>
        <div style='background:#1a3060;border:1px solid #fbbf2444;border-radius:8px;
                    padding:12px 24px;font-size:20px;font-weight:700;color:#fde68a;'>
          Z[1] = {output[1]:.4f}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    best = int(np.argmax(weights))
    st.markdown(f"""
    <div class='info-box' style='margin-top:12px;'>
    📖 <b>최종 해석</b>: "{TOKENS[query_idx]}"의 출력 Z는 모든 V를 attention weight로 가중합산한 결과입니다.
    이 벡터는 해당 토큰이 <b style='color:{TOKEN_COLORS[query_idx]};'>문장 내 다른 토큰과의 관계를 반영한 새로운 표현</b>입니다.<br>
    가장 많이 주목한 토큰:
    <b style='color:{TOKEN_COLORS[best]};'>{TOKENS[best]} ({weights[best]*100:.1f}%)</b>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;font-size:10px;color:#334;padding:10px 0;'>
  Scaled Dot-Product Attention · Vaswani et al. (2017) "Attention is All You Need"<br>
  dim=4 임베딩, dim_k=2 투영, 4개 토큰 한국어 예제
</div>
""", unsafe_allow_html=True)