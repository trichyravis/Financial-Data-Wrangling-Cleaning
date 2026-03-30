"""
=============================================================================
THE MOUNTAIN PATH - WORLD OF FINANCE
Benford's Law, Fraud Analytics & Anomaly Detection
Interactive Learning Platform
Prof. V. Ravichandran | themountainpathacademy.com
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Benford's Law | The Mountain Path",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
DARK_BLUE  = "#003366"
MID_BLUE   = "#004d80"
GOLD       = "#FFD700"
CARD_BG    = "#112240"
TXT        = "#e6f1ff"
MUTED      = "#8892b0"
GREEN      = "#28a745"
RED        = "#dc3545"
LIGHT_BLUE = "#ADD8E6"
BG_GRAD    = "linear-gradient(135deg,#1a2332,#243447,#2a3f5f)"

st.markdown(f"""
<style>
  /* ── base ── */
  .stApp {{
    background: {BG_GRAD};
    color: {TXT};
    font-family: 'Segoe UI', Arial, sans-serif;
  }}
  section[data-testid="stSidebar"] {{
    background: {CARD_BG} !important;
    border-right: 2px solid {GOLD};
  }}
  /* ── headings ── */
  h1,h2,h3,h4 {{ color:{GOLD} !important; }}
  /* ── metric ── */
  [data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid {GOLD}44;
    border-radius: 10px;
    padding: 12px;
  }}
  /* ── info/success/warning boxes ── */
  .stAlert {{ border-radius: 8px !important; }}
  /* ── tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {CARD_BG};
    border-radius: 8px 8px 0 0;
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {MUTED};
    font-weight: 600;
    border-radius: 6px 6px 0 0;
    padding: 8px 18px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {DARK_BLUE} !important;
    color: {GOLD} !important;
    border-bottom: 3px solid {GOLD};
  }}
  /* ── cards ── */
  .mp-card {{
    background: {CARD_BG};
    border: 1px solid {GOLD}44;
    border-left: 4px solid {GOLD};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .mp-card-red {{
    background: {CARD_BG};
    border: 1px solid {RED}66;
    border-left: 4px solid {RED};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .mp-card-green {{
    background: {CARD_BG};
    border: 1px solid {GREEN}66;
    border-left: 4px solid {GREEN};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .mp-card-blue {{
    background: {CARD_BG};
    border: 1px solid {LIGHT_BLUE}66;
    border-left: 4px solid {LIGHT_BLUE};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  /* ── hero ── */
  .hero-wrap {{
    background: linear-gradient(135deg,{DARK_BLUE},{MID_BLUE});
    border: 2px solid {GOLD};
    border-radius: 14px;
    padding: 30px 36px;
    text-align: center;
    margin-bottom: 24px;
  }}
  /* ── badge ── */
  .badge {{
    display: inline-block;
    background: {GOLD};
    color: {DARK_BLUE};
    font-weight: 700;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
    user-select: none;
  }}
  .badge-red {{
    background: {RED};
    color: white;
  }}
  .badge-green {{
    background: {GREEN};
    color: white;
  }}
  /* ── formula box ── */
  .formula-box {{
    background: {DARK_BLUE};
    border: 2px solid {GOLD};
    border-radius: 10px;
    padding: 18px 24px;
    text-align: center;
    margin: 16px 0;
    user-select: none;
  }}
  /* ── verdict banners ── */
  .verdict-ok {{
    background: linear-gradient(90deg,#1a3a1a,{CARD_BG});
    border-left: 5px solid {GREEN};
    border-radius: 8px;
    padding: 14px 20px;
    color: #aaffaa;
    font-weight: 700;
    font-size: 15px;
  }}
  .verdict-warn {{
    background: linear-gradient(90deg,#3a2a00,{CARD_BG});
    border-left: 5px solid {GOLD};
    border-radius: 8px;
    padding: 14px 20px;
    color: {GOLD};
    font-weight: 700;
    font-size: 15px;
  }}
  .verdict-bad {{
    background: linear-gradient(90deg,#3a0a0a,{CARD_BG});
    border-left: 5px solid {RED};
    border-radius: 8px;
    padding: 14px 20px;
    color: #ffaaaa;
    font-weight: 700;
    font-size: 15px;
  }}
  /* ── sidebar links ── */
  a {{ color:{GOLD} !important; text-decoration:none; }}
  a:hover {{ color:{LIGHT_BLUE} !important; text-decoration:underline; }}
  /* ── dataframe ── */
  [data-testid="stDataFrame"] {{ border-radius: 8px; }}
  /* ── slider ── */
  .stSlider [data-baseweb="slider"] {{ color:{GOLD}; }}
  /* suppress default stMarkdown padding */
  .block-container {{ padding-top: 1.5rem; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# BENFORD ENGINE
# ─────────────────────────────────────────────────────────────
BENFORD = {d: np.log10(1 + 1/d) for d in range(1, 10)}

def first_digit(x):
    x = abs(float(x))
    if x <= 0: return None
    s = f"{x:.10e}"
    return int(s[0])

def extract_digits(arr):
    digits = [first_digit(x) for x in arr if x > 0]
    return [d for d in digits if d is not None]

def benford_analysis(data):
    data = np.array(data, dtype=float)
    data = data[data > 0]
    n = len(data)
    if n == 0:
        return None
    digits = extract_digits(data)
    counts = {d: digits.count(d) for d in range(1, 10)}
    obs_p  = {d: counts[d] / n for d in range(1, 10)}
    exp_p  = BENFORD.copy()
    expected = {d: exp_p[d] * n for d in range(1, 10)}

    # Chi-squared
    chi2 = sum((counts[d] - expected[d])**2 / expected[d] for d in range(1, 10))
    p_chi2 = 1 - stats.chi2.cdf(chi2, df=8)

    # MAD
    mad = sum(abs(obs_p[d] - exp_p[d]) for d in range(1, 10)) / 9

    # Z-scores
    z = {}
    for d in range(1, 10):
        p_b = exp_p[d]
        se = np.sqrt(p_b * (1 - p_b) / n)
        z[d] = (abs(obs_p[d] - p_b) - 1/(2*n)) / se if se > 0 else 0

    # KS
    cdf_obs = np.cumsum([obs_p[d] for d in range(1, 10)])
    cdf_ben = np.cumsum([exp_p[d] for d in range(1, 10)])
    ks_stat = float(np.max(np.abs(cdf_obs - cdf_ben)))

    # Verdict
    if mad < 0.006:
        verdict, level = "CLOSE CONFORMITY — No significant anomaly", "ok"
    elif mad < 0.012:
        verdict, level = "ACCEPTABLE CONFORMITY — Monitor", "warn"
    elif mad < 0.015:
        verdict, level = "MARGINAL CONFORMITY — Review Recommended", "warn"
    else:
        verdict, level = "NON-CONFORMING — INVESTIGATE IMMEDIATELY", "bad"

    return dict(n=n, counts=counts, obs_p=obs_p, exp_p=exp_p,
                expected=expected, chi2=chi2, p_chi2=p_chi2,
                mad=mad, z=z, ks=ks_stat, verdict=verdict, level=level,
                digits=digits)

def last_digit_analysis(data):
    data = np.array(data, dtype=float)
    data = data[data > 0]
    last_digits = [int(str(int(abs(x)))[-1]) for x in data if x >= 1]
    counts = {d: last_digits.count(d) for d in range(10)}
    n = len(last_digits)
    probs = {d: counts[d]/n for d in range(10)} if n > 0 else {}
    round_pct = (counts.get(0, 0) + counts.get(5, 0)) / n * 100 if n > 0 else 0
    return counts, probs, round_pct

# ─────────────────────────────────────────────────────────────
# CHART FACTORY
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=CARD_BG,
    font=dict(color=TXT, family="Segoe UI, Arial"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def chart_first_digit(result, title="Benford Analysis"):
    digits = list(range(1, 10))
    obs = [result["obs_p"][d]*100 for d in digits]
    ben = [result["exp_p"][d]*100 for d in digits]
    z_vals = [result["z"][d] for d in digits]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["First-Digit Frequency vs Benford's Law",
                        "Z-Score per Digit (Statistical Significance)"],
        horizontal_spacing=0.12,
    )

    # Bar chart
    fig.add_trace(go.Bar(
        x=[str(d) for d in digits], y=obs,
        name="Observed", marker_color=DARK_BLUE,
        marker_line=dict(color=GOLD, width=1.2),
        hovertemplate="Digit %{x}<br>Observed: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[str(d) for d in digits], y=ben,
        name="Benford's Law", marker_color=GOLD,
        marker_opacity=0.85,
        hovertemplate="Digit %{x}<br>Expected: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[str(d) for d in digits], y=ben,
        mode="lines+markers", name="Benford Curve",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(symbol="circle-open", size=7, color="white"),
        showlegend=False,
    ), row=1, col=1)

    # Z-score bar
    z_colors = [RED if z > 2.576 else "#fd7e14" if z > 1.96 else DARK_BLUE
                for z in z_vals]
    fig.add_trace(go.Bar(
        x=[str(d) for d in digits], y=z_vals,
        name="Z-Score", marker_color=z_colors,
        hovertemplate="Digit %{x}<br>Z = %{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    for threshold, color, label in [(1.96, "#fd7e14", "5% sig"),
                                     (2.576, RED, "1% sig")]:
        fig.add_hline(y=threshold, line_dash="dash", line_color=color,
                      annotation_text=label, row=1, col=2,
                      annotation_font_color=color)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"<b>{title}</b>", font=dict(color=GOLD, size=14)),
        barmode="group", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.25,
                    font=dict(color=TXT)),
    )
    fig.update_xaxes(title_text="Leading Digit", title_font_color=MUTED,
                     gridcolor="#2a3f5f", linecolor="#2a3f5f")
    fig.update_yaxes(title_font_color=MUTED, gridcolor="#2a3f5f",
                     linecolor="#2a3f5f")
    return fig


def chart_deviation(result):
    digits = list(range(1, 10))
    devs = [(result["obs_p"][d] - result["exp_p"][d])*100 for d in digits]
    colors = [RED if d > 1.5 else GREEN if d < -1.5 else MID_BLUE for d in devs]
    fig = go.Figure(go.Bar(
        x=[str(d) for d in digits], y=devs,
        marker_color=colors,
        hovertemplate="Digit %{x}<br>Deviation: %{y:+.2f}%<extra></extra>",
        text=[f"{v:+.1f}%" for v in devs],
        textposition="outside",
        textfont=dict(color=TXT, size=11),
    ))
    fig.add_hline(y=0, line_color="white", line_width=1.5)
    fig.add_hline(y=1.5,  line_dash="dash", line_color=RED, line_width=1,
                  annotation_text="Alert +1.5%", annotation_font_color=RED)
    fig.add_hline(y=-1.5, line_dash="dash", line_color=GOLD, line_width=1,
                  annotation_text="Alert -1.5%", annotation_font_color=GOLD)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>Deviation from Benford's Law (%)</b>",
                   font=dict(color=GOLD, size=14)),
        xaxis_title="Leading Digit",
        yaxis_title="Observed − Expected (%)",
        height=360,
    )
    return fig


def chart_last_digit(probs):
    digits = list(range(10))
    vals = [probs.get(d, 0)*100 for d in digits]
    colors = [RED if d in (0, 5) else DARK_BLUE for d in digits]
    fig = go.Figure(go.Bar(
        x=[str(d) for d in digits], y=vals,
        marker_color=colors,
        hovertemplate="Last digit %{x}<br>%{y:.1f}%<extra></extra>",
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color=TXT, size=10),
    ))
    fig.add_hline(y=10, line_dash="dash", line_color=GOLD,
                  annotation_text="Expected 10%",
                  annotation_font_color=GOLD)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>Last-Digit Distribution (Round-Number Test)</b>",
                   font=dict(color=GOLD, size=14)),
        xaxis_title="Last Digit",
        yaxis_title="Frequency (%)",
        height=360,
    )
    return fig


def chart_log_histogram(data):
    data = np.array(data, dtype=float)
    data = data[data > 0]
    log_data = np.log10(data)
    fig = go.Figure(go.Histogram(
        x=log_data, nbinsx=60,
        marker_color=MID_BLUE,
        marker_line=dict(color=DARK_BLUE, width=0.3),
        hovertemplate="log₁₀(Amount): %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    for d in range(1, 10):
        fig.add_vline(x=np.log10(d), line_color=GOLD,
                      line_width=0.6, opacity=0.5)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>Transaction Amount Distribution (Log₁₀ Scale)</b>",
                   font=dict(color=GOLD, size=14)),
        xaxis_title="log₁₀(Amount) — Gold lines mark digit boundaries",
        yaxis_title="Count",
        height=350,
    )
    return fig

# ─────────────────────────────────────────────────────────────
# HELPER: display result panel
# ─────────────────────────────────────────────────────────────
def show_stats_panel(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample Size",  f"{result['n']:,}")
    c2.metric("Chi-Squared",  f"{result['chi2']:.2f}",
              delta=f"p = {result['p_chi2']:.4f}",
              delta_color="inverse")
    c3.metric("MAD",          f"{result['mad']:.4f}",
              delta="Threshold: 0.015", delta_color="off")
    c4.metric("KS Statistic", f"{result['ks']:.4f}")

    css = {"ok":"verdict-ok","warn":"verdict-warn","bad":"verdict-bad"}
    st.markdown(
        f'<div class="{css[result["level"]]}">'
        f'🔍 {result["verdict"]}</div>', unsafe_allow_html=True
    )

def show_digit_table(result):
    rows = []
    for d in range(1, 10):
        z = result["z"][d]
        flag = "🚨 ***" if z > 2.576 else "⚠️ **" if z > 1.96 else \
               "⚡ *" if z > 1.645 else "✅"
        rows.append({
            "Digit": d,
            "Observed": result["counts"][d],
            "Expected": f"{result['expected'][d]:.1f}",
            "Obs %": f"{result['obs_p'][d]*100:.2f}%",
            "Benford %": f"{result['exp_p'][d]*100:.2f}%",
            "Deviation": f"{(result['obs_p'][d]-result['exp_p'][d])*100:+.2f}%",
            "Z-Score": f"{z:.3f}",
            "Flag": flag,
        })
    df = pd.DataFrame(rows).set_index("Digit")
    st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:16px 0 8px;">
      <div style="font-size:26px; color:{GOLD}; font-weight:900; letter-spacing:1px;">
        🏔️ THE MOUNTAIN PATH
      </div>
      <div style="font-size:12px; color:{MUTED}; margin-top:4px;">
        World of Finance
      </div>
      <div style="height:2px; background:{GOLD}; margin:10px 0;"></div>
      <div style="font-size:11px; color:{LIGHT_BLUE};">
        Prof. V. Ravichandran<br>
        <a href="https://themountainpathacademy.com" target="_blank">
          themountainpathacademy.com
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "📚 Navigate",
        ["🏠 Home",
         "📖 Learn: Benford's Law",
         "📊 Interactive Analyzer",
         "🏦 Case 1: GST Invoice Fraud",
         "💼 Case 2: Expense Report Fraud",
         "🏛️ Case 3: Bank Structuring",
         "🤖 ML Anomaly Detection",
         "❓ Quiz & Assessment"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px; color:{MUTED}; line-height:1.7;">
      <b style="color:{GOLD};">Key Topics</b><br>
      • Benford's Law Formula<br>
      • First & Second Digit Tests<br>
      • Chi-Squared & MAD Tests<br>
      • GST / Tax Fraud Detection<br>
      • Expense Reimbursement Fraud<br>
      • PMLA Structuring Detection<br>
      • Isolation Forest ML<br>
      • Ethical Considerations
    </div>
    <div style="height:2px; background:{GOLD}22; margin:12px 0;"></div>
    <div style="font-size:10px; color:{MUTED}; text-align:center;">
      © 2025 The Mountain Path<br>
      <a href="https://www.linkedin.com/in/trichyravis" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
      <a href="https://github.com/trichyravis" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown(f"""
    <div class="hero-wrap">
      <div style="font-size:42px; color:{GOLD}; font-weight:900; letter-spacing:2px;">
        BENFORD'S LAW
      </div>
      <div style="font-size:22px; color:{TXT}; margin:6px 0;">
        Fraud Analytics &amp; Anomaly Detection
      </div>
      <div style="font-size:14px; color:{MUTED}; margin-top:10px;">
        An Interactive Learning Platform &nbsp;|&nbsp; The Mountain Path – World of Finance
      </div>
      <div style="margin-top:14px;">
        <span class="badge">Financial Analytics</span>
        <span class="badge">Fraud Detection</span>
        <span class="badge">Statistical Testing</span>
        <span class="badge">Machine Learning</span>
        <span class="badge">Python</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, "📐", "P(d) = log₁₀(1+1/d)", "Benford Formula"),
        (c2, "📊", "30.1%", "Numbers Starting with 1"),
        (c3, "🔍", "3 Live Caselets", "Fraud Scenarios"),
        (c4, "🤖", "Isolation Forest", "ML Anomaly Detection"),
    ]:
        col.metric(label, val)

    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📌 What You Will Learn")
        for item in [
            ("📐 The Mathematics", "Understand Benford's Law from first principles with the logarithmic intuition"),
            ("📊 Statistical Tests", "Chi-squared, MAD, Z-score and KS tests with worked examples"),
            ("🏦 GST Fraud", "Detect fictitious invoice fraud in GST returns using digit analysis"),
            ("💼 Expense Fraud", "Identify inflated and fabricated corporate expense claims"),
            ("🏛️ Bank Structuring", "Catch PMLA threshold gaming in banking transactions"),
            ("🤖 ML Methods", "Isolation Forest for multivariate transaction anomaly detection"),
            ("⚖️ Ethics", "Responsible use of fraud analytics in practice"),
        ]:
            st.markdown(f"""
            <div class="mp-card">
              <b style="color:{GOLD};">{item[0]}</b>
              <span style="color:{TXT}; margin-left:8px;">{item[1]}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Benford's Law at a Glance")
        digits = list(range(1, 10))
        probs  = [BENFORD[d]*100 for d in digits]
        fig = go.Figure(go.Bar(
            x=[str(d) for d in digits], y=probs,
            marker_color=[DARK_BLUE if i < 3 else MID_BLUE for i in range(9)],
            marker_line=dict(color=GOLD, width=1),
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(color=TXT, size=10),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="<b>Expected First-Digit Frequency</b>",
                       font=dict(color=GOLD, size=13)),
            xaxis_title="Leading Digit",
            yaxis_title="Probability (%)",
            height=340,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="formula-box">
          <div style="color:{MUTED}; font-size:11px; margin-bottom:6px;">
            BENFORD'S LAW FORMULA
          </div>
          <div style="color:{GOLD}; font-size:24px; font-weight:900; font-family:Georgia,serif;">
            P(d) = log<sub>10</sub>(1 + 1/d)
          </div>
          <div style="color:{LIGHT_BLUE}; font-size:11px; margin-top:6px;">
            d ∈ {{1, 2, 3, 4, 5, 6, 7, 8, 9}}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("""
    **How to Use This Platform:** Use the sidebar navigation to move between sections.
    Start with *Learn: Benford's Law* for the theory, then explore the three live caselets,
    the interactive analyzer, and finally the ML anomaly detection module.
    Each section builds on the previous one.
    """)

# ─────────────────────────────────────────────────────────────
# PAGE: LEARN
# ─────────────────────────────────────────────────────────────
elif page == "📖 Learn: Benford's Law":
    st.markdown(f"## 📖 Understanding Benford's Law")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📜 History & Concept",
        "📐 Mathematics",
        "📊 Extended Analysis",
        "🧪 Statistical Tests",
        "⚠️ Limitations & Ethics",
    ])

    # ── TAB 1: History
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🕰️ A Discovery from a Worn Logarithm Book")
            st.markdown(f"""
            <div class="mp-card">
              <b style="color:{GOLD};">1881 — Simon Newcomb</b><br>
              An American astronomer noticed the early pages of logarithm tables were
              significantly more worn than the later pages — people looked up numbers
              starting with 1 far more often. He published this observation, but it
              received little attention.
            </div>
            <div class="mp-card">
              <b style="color:{GOLD};">1938 — Frank Benford</b><br>
              Physicist Frank Benford rediscovered the phenomenon and systematically tested
              <b>20,229 data points</b> from 20 different datasets: river surface areas,
              atomic weights, baseball statistics, street addresses, population figures,
              and more. The same pattern appeared every time.
            </div>
            <div class="mp-card-green">
              <b style="color:{GREEN};">The Core Insight</b><br>
              In many naturally occurring datasets, the leading (first) digit is <em>not</em>
              uniformly distributed. The digit 1 appears about 30% of the time — six times
              more often than the digit 9 (≈4.6%). This is the "First-Digit Law."
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📋 When Does Benford's Law Apply?")
            col_a, col_b = st.columns(2)
            with col_a:
                st.success("✅ **Follows Benford's Law**")
                items = [
                    "Revenue and sales figures",
                    "Expense and invoice amounts",
                    "Tax return values",
                    "Financial statement line items",
                    "Stock prices and volumes",
                    "Population data",
                    "Scientific measurements",
                    "GST and customs values",
                ]
                for i in items:
                    st.markdown(f"• {i}")
            with col_b:
                st.error("❌ **Does NOT Follow Benford's Law**")
                items2 = [
                    "Phone numbers (fixed format)",
                    "Employee / Account IDs (sequential)",
                    "Ages (bounded range 0-120)",
                    "Prices with psychological anchoring (₹999)",
                    "Numbers assigned uniformly",
                    "Lottery numbers",
                    "Zip/Postal codes",
                    "Data with min/max constraints",
                ]
                for i in items2:
                    st.markdown(f"• {i}")

        with col2:
            st.markdown("### 📊 The Benford Distribution")
            df_ben = pd.DataFrame({
                "Digit": range(1, 10),
                "Probability (%)": [round(BENFORD[d]*100, 2) for d in range(1,10)],
                "1-in-N": [f"1 in {round(1/BENFORD[d])}" for d in range(1,10)],
            })
            st.dataframe(df_ben.set_index("Digit"), use_container_width=True)

            st.markdown(f"""
            <div class="formula-box" style="margin-top:16px;">
              <div style="color:{MUTED}; font-size:11px;">KEY FACT</div>
              <div style="color:{GOLD}; font-size:18px; font-weight:800; margin:8px 0;">
                Digits 1–3 account for<br>
                <span style="font-size:36px;">60.2%</span><br>
                of all leading digits!
              </div>
              <div style="color:{LIGHT_BLUE}; font-size:11px;">
                While digit 9 appears only 4.6% of the time
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Mathematics
    with tab2:
        st.markdown("### 📐 The Mathematics of Benford's Law")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
              <div style="color:{MUTED}; font-size:12px; margin-bottom:8px;">
                BENFORD'S LAW — FIRST DIGIT
              </div>
              <div style="color:{GOLD}; font-size:28px; font-weight:900; font-family:Georgia,serif;">
                P(d) = log<sub>10</sub>(1 + 1/d)
              </div>
              <div style="color:{TXT}; font-size:13px; margin-top:10px;">
                where d ∈ {{1, 2, 3, 4, 5, 6, 7, 8, 9}}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### The Logarithmic Intuition")
            st.markdown(f"""
            <div class="mp-card-blue">
              <b style="color:{LIGHT_BLUE};">Step-by-step walkthrough with compound growth:</b><br><br>
              Start with ₹100 (first digit: <b style="color:{GOLD};">1</b>)<br>
              +10% → ₹110 → first digit: <b style="color:{GOLD};">1</b><br>
              +10% → ₹121 → first digit: <b style="color:{GOLD};">1</b><br>
              +10% → ₹133 → first digit: <b style="color:{GOLD};">1</b><br>
              +10% → ₹146 → first digit: <b style="color:{GOLD};">1</b><br>
              +10% → ₹161 → first digit: <b style="color:{GOLD};">1</b><br>
              +10% → ₹177 → first digit: <b style="color:{GOLD};">1</b><br>
              +10% → ₹195 → first digit: <b style="color:{GOLD};">1</b><br>
              +10% → ₹214 → first digit: <b style="color:{GOLD};">2</b> ← Finally leaves "1"!<br><br>
              <b>Key insight:</b> Numbers spend far more "time" starting with 1 than with 9.
              To move from 1... to 2... requires a <b>100% increase</b>.
              To move from 9... to 10... requires only an <b>11% increase</b>.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Interactive: explore individual digit probabilities
            st.markdown("#### 🔢 Explore Individual Digit Probabilities")
            sel_digit = st.slider("Select a leading digit:", 1, 9, 1)
            prob = BENFORD[sel_digit]
            st.markdown(f"""
            <div style="background:{CARD_BG}; border:2px solid {GOLD};
                border-radius:10px; padding:20px; text-align:center; margin:10px 0;">
              <div style="color:{MUTED}; font-size:12px;">Formula</div>
              <div style="color:{GOLD}; font-size:18px; font-family:Georgia,serif;">
                P({sel_digit}) = log₁₀(1 + 1/{sel_digit}) = log₁₀({sel_digit+1}/{sel_digit})
              </div>
              <div style="color:{TXT}; font-size:36px; font-weight:900; margin:12px 0;">
                {prob*100:.2f}%
              </div>
              <div style="color:{LIGHT_BLUE}; font-size:13px;">
                Appears approximately 1 in {round(1/prob)} numbers
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Compute and show cumulative
            st.markdown("#### 📈 Cumulative Distribution")
            cum_data = []
            running = 0
            for d in range(1, 10):
                running += BENFORD[d]*100
                cum_data.append({"Digit": d, "Cumulative %": round(running, 2)})
            st.dataframe(pd.DataFrame(cum_data).set_index("Digit"),
                         use_container_width=True)

    # ── TAB 3: Extended
    with tab3:
        st.markdown("### 📊 Extended Benford Analysis")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Second-Digit Distribution")
            st.markdown(f"""
            <div class="mp-card">
              Benford's Law extends to the second digit. The second digit can be 0–9
              and the distribution is more uniform than the first digit (ranging from
              ~12% for 0 down to ~8.5% for 9).
            </div>
            """, unsafe_allow_html=True)

            second_digit_probs = {
                d: sum(np.log10(1 + 1/(10*k + d)) for k in range(1, 10))
                for d in range(0, 10)
            }
            df_2nd = pd.DataFrame({
                "Digit": list(range(10)),
                "Expected %": [round(second_digit_probs[d]*100, 2) for d in range(10)],
            })
            st.dataframe(df_2nd.set_index("Digit"), use_container_width=True)

        with c2:
            st.markdown("#### 🔴 Round Number / Last-Digit Test")
            st.markdown(f"""
            <div class="mp-card-red">
              <b style="color:{RED};">Fraudsters Love Round Numbers!</b><br><br>
              When humans fabricate amounts they gravitate towards convenient values:
              ₹10,000; ₹25,000; ₹50,000; ₹1,00,000.<br><br>
              <b>Detection signal:</b> In natural data, each last digit (0–9) appears
              ~10% of the time. In fabricated data, digits 0 and 5 often appear 30-40%.<br><br>
              <b style="color:{GOLD};">Round Number Score = Observed % of 0s and 5s ÷ 20%</b><br>
              Any score > 2.0 (i.e., >40%) is a significant red flag.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📋 The Summation Test")
        st.markdown(f"""
        <div class="mp-card-blue">
          <b style="color:{LIGHT_BLUE};">What it detects:</b> Whether any specific two-digit combination
          accounts for a <em>disproportionate share of total transaction value</em>
          (indicating round-number or threshold-gaming fraud).<br><br>
          In a clean dataset, each two-digit group (10–99) should account for approximately
          <b style="color:{GOLD};">1/90 ≈ 1.11%</b> of total value.
          Any group with a ratio >2× the expected share is flagged for investigation.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: Statistical Tests
    with tab4:
        st.markdown("### 🧪 Statistical Tests for Benford's Law")

        st.markdown(f"""
        <div class="mp-card">
          <b style="color:{GOLD};">Why do we need statistical tests?</b><br>
          Observed frequencies will never perfectly match Benford's Law due to random
          sampling variation. Statistical tests tell us: <em>Is this deviation large enough
          to be suspicious, or could it be random chance?</em>
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["Chi-Squared", "MAD Test", "Z-Score", "KS Test"])

        with t1:
            st.markdown(f"""
            <div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">CHI-SQUARED GOODNESS-OF-FIT</div>
              <div style="color:{GOLD}; font-size:22px; font-family:Georgia,serif;">
                χ² = Σ (Oᵈ − Eᵈ)² / Eᵈ
              </div>
              <div style="color:{TXT}; font-size:12px; margin-top:8px;">
                Degrees of freedom = 8 (nine digits minus one constraint)
              </div>
            </div>
            """, unsafe_allow_html=True)
            df_crit = pd.DataFrame({
                "Significance Level": ["10%", "5%", "1%", "0.1%"],
                "Critical Value (df=8)": [13.36, 15.51, 20.09, 26.12],
                "Action if Exceeded": ["Review", "Investigate", "Urgent Investigation", "Immediate Alert"],
            })
            st.dataframe(df_crit, use_container_width=True, hide_index=True)
            st.warning("⚠️ **Limitation:** With large samples (n > 5,000), even tiny, practically insignificant deviations become statistically significant. Always combine with MAD.")

        with t2:
            st.markdown(f"""
            <div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">MEAN ABSOLUTE DEVIATION (MAD)</div>
              <div style="color:{GOLD}; font-size:22px; font-family:Georgia,serif;">
                MAD = (1/9) × Σ |Pᵈobs − Pᵈbenford|
              </div>
            </div>
            """, unsafe_allow_html=True)
            df_mad = pd.DataFrame({
                "MAD Value": ["0.000 – 0.006", "0.006 – 0.012", "0.012 – 0.015", "> 0.015"],
                "Interpretation": ["Close conformity", "Acceptable conformity", "Marginal conformity", "Non-conformity"],
                "Action": ["No issue ✅", "Monitor 👀", "Review ⚠️", "INVESTIGATE 🚨"],
            })
            st.dataframe(df_mad, use_container_width=True, hide_index=True)
            st.success("✅ **Advantage:** MAD is not inflated by large sample sizes — the preferred metric in forensic accounting (Nigrini, 2012).")

        with t3:
            st.markdown(f"""
            <div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">Z-SCORE PER DIGIT</div>
              <div style="color:{GOLD}; font-size:18px; font-family:Georgia,serif;">
                Zᵈ = (|Pᵈobs − Pᵈbenford| − 1/(2n)) / √(Pᵈbenford(1−Pᵈbenford)/n)
              </div>
              <div style="color:{TXT}; font-size:11px; margin-top:6px;">
                Flag: |Z| > 1.96 (5%) or |Z| > 2.576 (1%)
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 **Use case:** Identifies WHICH specific digit is anomalous, guiding the fraud investigator to the exact pattern.")

        with t4:
            st.markdown(f"""
            <div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">KOLMOGOROV-SMIRNOV TEST</div>
              <div style="color:{GOLD}; font-size:22px; font-family:Georgia,serif;">
                D = max |Fobs(d) − FBenford(d)|
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 **Best for:** Small samples (< 100 records) where chi-squared assumptions may not hold.")

    # ── TAB 5: Limitations
    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚠️ Critical Limitations")
            limits = [
                ("Not all data qualifies", "Phone numbers, ages, psychologically-anchored prices do not follow Benford. Always verify applicability before testing."),
                ("Sample size sensitivity", "Chi-squared becomes overly sensitive with n > 5,000. Use MAD as the primary metric for large datasets."),
                ("Benford conformity ≠ innocence", "A sophisticated fraudster who knows the law can deliberately fabricate Benford-conforming numbers."),
                ("Industry-specific deviations", "FMCG companies price goods at ₹X9.99; real estate near stamp duty thresholds. Benchmark against peers."),
                ("Minimum sample size needed", "At least 50–100 records for any inference; ideally 1,000+."),
            ]
            for title, desc in limits:
                st.markdown(f"""
                <div class="mp-card-red">
                  <b style="color:{RED};">{title}</b><br>
                  <span style="color:{TXT};">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### ⚖️ Ethical Principles")
            ethics = [
                ("Presumption of innocence", "A statistical flag is NOT evidence of fraud. It is a signal for investigation. Verify thoroughly before any action."),
                ("Algorithmic fairness", "Audit ML models for demographic bias. Disproportionate flagging of any group signals a fairness problem."),
                ("Data privacy & proportionality", "Transaction monitoring must comply with DPDP Act 2023 (India) and be proportionate to risk."),
                ("Transparency", "Individuals subject to adverse decisions have the right to understand and contest the reason."),
                ("Human-in-the-loop", "Machines flag; humans decide. High-stakes actions (account freezing, FIU reporting) require human review."),
                ("Model governance", "Fraud models must be re-validated periodically. Fraudsters adapt; static models become obsolete."),
            ]
            for title, desc in ethics:
                st.markdown(f"""
                <div class="mp-card-green">
                  <b style="color:{GREEN};">{title}</b><br>
                  <span style="color:{TXT};">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: INTERACTIVE ANALYZER
# ─────────────────────────────────────────────────────────────
elif page == "📊 Interactive Analyzer":
    st.markdown("## 📊 Interactive Benford Analyzer")
    st.markdown("Upload your own data or use the synthetic generator to explore Benford's Law in action.")

    tab_upload, tab_gen = st.tabs(["📁 Upload / Paste Data", "🎲 Synthetic Data Generator"])

    with tab_upload:
        col1, col2 = st.columns([1, 2])
        with col1:
            method = st.radio("Input method:", ["Paste numbers", "Upload CSV"])
            if method == "Paste numbers":
                raw = st.text_area("Paste numbers (one per line or comma-separated):",
                    "145230\n23400\n8750\n312000\n67800\n190000\n445000\n28900\n"
                    "73500\n156000\n92000\n34500\n228000\n87600\n143000",
                    height=200)
                if st.button("Analyse", type="primary"):
                    try:
                        nums = []
                        for token in raw.replace(",", "\n").split():
                            try: nums.append(float(token))
                            except: pass
                        st.session_state["analyzer_data"] = nums
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                uploaded = st.file_uploader("Upload CSV file:", type=["csv"])
                if uploaded:
                    df_up = pd.read_csv(uploaded)
                    numeric_cols = df_up.select_dtypes(include=np.number).columns.tolist()
                    col_sel = st.selectbox("Select numeric column:", numeric_cols)
                    if st.button("Analyse Column", type="primary"):
                        st.session_state["analyzer_data"] = df_up[col_sel].dropna().tolist()

        with col2:
            if "analyzer_data" in st.session_state:
                data = st.session_state["analyzer_data"]
                result = benford_analysis(data)
                if result:
                    show_stats_panel(result)
                    st.plotly_chart(chart_first_digit(result, "Your Data — Benford Analysis"),
                                    use_container_width=True)
                    show_digit_table(result)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(chart_deviation(result), use_container_width=True)
                    with c2:
                        last_counts, last_probs, round_pct = last_digit_analysis(data)
                        st.plotly_chart(chart_last_digit(last_probs), use_container_width=True)
                        if round_pct > 30:
                            st.error(f"🚨 Round Number Score: {round_pct:.1f}% of last digits are 0 or 5 (expected: 20%). Suspicious!")
                        else:
                            st.success(f"✅ Round Number Score: {round_pct:.1f}% (normal range)")
                    st.plotly_chart(chart_log_histogram(data), use_container_width=True)

    with tab_gen:
        st.markdown("### 🎲 Synthetic Data Generator")
        col1, col2 = st.columns([1, 2])
        with col1:
            n_clean = st.slider("Clean transactions:", 500, 5000, 2000, 100)
            n_fraud = st.slider("Fraudulent transactions:", 0, 500, 150, 10)
            fraud_type = st.selectbox("Fraud pattern:",
                ["Structuring (just-below threshold)",
                 "Round numbers (₹10k, ₹25k, ₹50k)",
                 "Digit-5 inflation",
                 "No fraud (clean dataset)"])

            if st.button("Generate & Analyse", type="primary"):
                np.random.seed(42)
                clean = np.concatenate([
                    10**np.random.uniform(2, 6, int(n_clean*0.7)),
                    np.random.exponential(25000, int(n_clean*0.3))
                ])
                if fraud_type == "No fraud (clean dataset)":
                    synth = clean
                elif fraud_type == "Structuring (just-below threshold)":
                    fraud = np.random.uniform(950000, 999000, n_fraud)
                    synth = np.concatenate([clean, fraud])
                elif fraud_type == "Round numbers (₹10k, ₹25k, ₹50k)":
                    fraud = np.random.choice([10000, 25000, 50000, 100000], n_fraud)
                    synth = np.concatenate([clean, fraud])
                else:
                    fraud = np.random.uniform(50000, 59999, n_fraud)
                    synth = np.concatenate([clean, fraud])

                st.session_state["synth_data"] = synth
                st.session_state["synth_type"] = fraud_type

        with col2:
            if "synth_data" in st.session_state:
                result = benford_analysis(st.session_state["synth_data"])
                if result:
                    fraud_type_lbl = st.session_state.get("synth_type", "")
                    st.caption(f"📊 Analysis: {fraud_type_lbl}")
                    show_stats_panel(result)
                    st.plotly_chart(
                        chart_first_digit(result, f"Synthetic: {fraud_type_lbl}"),
                        use_container_width=True
                    )
                    show_digit_table(result)

# ─────────────────────────────────────────────────────────────
# CASE 1: GST INVOICE FRAUD
# ─────────────────────────────────────────────────────────────
elif page == "🏦 Case 1: GST Invoice Fraud":
    st.markdown("## 🏦 Case Study 1: GST Invoice Fraud Detection")

    # Scenario
    st.markdown(f"""
    <div class="hero-wrap" style="padding:20px 28px; text-align:left;">
      <span class="badge">Case Study 1</span>
      <span class="badge-red badge">FRAUD DETECTED</span>
      <h3 style="color:{GOLD}; margin:10px 0 6px;">Fictitious Invoice Fraud in GST Returns</h3>
      <p style="color:{TXT}; margin:0;">
        <b>Scenario:</b> A textile trader in Mumbai files GST returns claiming ₹5 crore in
        Input Tax Credit (ITC) based on 200 purchase invoices from three suppliers.
        The GST Network (GSTN) analytics team runs a Benford's Law test on this trader's
        invoice amounts as part of routine screening.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Claimed ITC", "₹5,00,00,000", delta="Fraudulent")
    col2.metric("Invoices Scrutinised", "200")
    col3.metric("Suppliers Involved", "3 (shell entities)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Background",
        "📊 Benford Analysis",
        "🔍 Investigation Findings",
        "📚 Learning Points",
    ])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""
            <div class="mp-card">
              <b style="color:{GOLD};">Business Context</b><br><br>
              M/s Radha Textile Traders filed GSTR-3B returns for FY2023-24 claiming
              substantial ITC on raw material purchases. A data-driven risk-scoring system
              flagged this taxpayer for having:
              <ul style="margin:8px 0; padding-left:20px;">
                <li>ITC-to-turnover ratio 340% higher than industry average</li>
                <li>All three suppliers registered within the past 6 months</li>
                <li>Invoice amounts unusually concentrated in specific ranges</li>
              </ul>
            </div>
            <div class="mp-card-red">
              <b style="color:{RED};">Known Fraud Behaviour</b><br><br>
              When humans fabricate invoice amounts they tend to:
              <ul style="margin:8px 0; padding-left:20px;">
                <li>Use "round" amounts (₹25,000; ₹50,000; ₹1,00,000)</li>
                <li>Cluster near but below reporting thresholds</li>
                <li>Underuse amounts beginning with 1 ("too small looking")</li>
                <li>Overuse amounts beginning with 5–9 ("looks more significant")</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            # Simulated invoice data
            np.random.seed(101)
            # Fraudulent: heavily weighted toward digits 5, 9 and round numbers
            fraud_invoices = np.concatenate([
                np.random.uniform(50000, 59999, 80),    # digit-5
                np.random.uniform(90000, 99000, 60),    # digit-9 structuring
                np.random.choice([25000, 50000, 100000, 200000], 40),  # round
                np.random.uniform(10000, 19999, 20),    # some digit-1
            ])
            st.session_state["case1_data"] = fraud_invoices

            st.markdown("**Sample Invoice Amounts (₹):**")
            sample_df = pd.DataFrame({
                "Invoice #": [f"INV-{i+1:03d}" for i in range(15)],
                "Amount (₹)": [f"₹{x:,.0f}" for x in fraud_invoices[:15]],
                "Supplier": np.random.choice(["Shree Fab", "Aarav Traders", "Lotus Tex"], 15),
            })
            st.dataframe(sample_df, use_container_width=True, hide_index=True)

    with tab2:
        data = st.session_state.get("case1_data",
            np.concatenate([
                np.random.uniform(50000, 59999, 80),
                np.random.uniform(90000, 99000, 60),
                np.random.choice([25000, 50000, 100000], 40),
                np.random.uniform(10000, 19999, 20),
            ]))
        result = benford_analysis(data)

        st.markdown("### Benford Analysis Results")
        show_stats_panel(result)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_first_digit(result, "GST Invoice Benford Analysis"),
                            use_container_width=True)
        with col2:
            st.plotly_chart(chart_deviation(result), use_container_width=True)

        show_digit_table(result)

        last_counts, last_probs, round_pct = last_digit_analysis(data)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_last_digit(last_probs), use_container_width=True)
        with col2:
            st.markdown(f"""
            <div class="mp-card-red" style="margin-top:20px;">
              <b style="color:{RED}; font-size:16px;">🚨 Round Number Alert</b><br><br>
              <b style="color:{GOLD}; font-size:24px;">{round_pct:.1f}%</b>
              of invoice amounts end in 0 or 5<br>
              <span style="color:{MUTED};">(Expected: ~20% in natural data)</span><br><br>
              Round Number Score: <b style="color:{RED};">{round_pct/20:.1f}×</b> the expected value
              <br><br>
              <b>This confirms fabricated amounts.</b> Real supplier invoices for
              raw materials would have irregular last digits reflecting actual weights,
              rates, and quantity calculations.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 🔍 Investigation Findings")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="mp-card-red">
              <b style="color:{RED};">Benford Signals That Triggered Investigation</b>
              <ul style="margin:10px 0; padding-left:20px; color:{TXT};">
                <li><b>Digit-1 frequency: 10%</b> (expected: 30.1%) — massive under-representation</li>
                <li><b>Digit-5 frequency: 40%</b> (expected: 7.9%) — massive over-representation</li>
                <li><b>Digit-9 frequency: 30%</b> (expected: 4.6%) — extreme structuring signal</li>
                <li><b>MAD = {result['mad']:.4f}</b> — far above 0.015 threshold</li>
                <li><b>χ² = {result['chi2']:.1f}</b> — overwhelmingly rejects Benford conformity</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="mp-card-red">
              <b style="color:{RED};">What Officers Found on Physical Verification</b>
              <ul style="margin:10px 0; padding-left:20px; color:{TXT};">
                <li>All three "suppliers" had the same registered address (a vacant plot)</li>
                <li>No physical inventory movement; no e-way bills matched</li>
                <li>Bank accounts of suppliers linked to same beneficial owner</li>
                <li>All supplier GST registrations cancelled for non-filing within 8 months</li>
                <li>₹5 Cr ITC fraud confirmed; criminal prosecution initiated</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="mp-card-green">
              <b style="color:{GREEN};">What Legitimate Textile Invoices Look Like</b><br><br>
              Real purchase invoices in the textile trade typically show:<br><br>
              • Amounts calculated as: <i>Qty × Rate × (1 + GST%)</i><br>
              • Rates like ₹145.50/metre, ₹287.25/kg — creating non-round amounts<br>
              • Natural variation across the full digit range<br>
              • MAD < 0.010 (within acceptable conformity)<br><br>
              <b>Benford's Law acts as an instant filter</b> — one test on 200 invoices
              immediately flagged what would otherwise require weeks of manual audit.
            </div>
            """, unsafe_allow_html=True)

            outcomes = {
                "Metric": ["Detection time", "ITC demand raised", "Penalty (200%)", "Criminal referral", "Suppliers blacklisted"],
                "Outcome": ["< 2 hours (analytics)", "₹5,00,00,000", "₹10,00,00,000", "Yes — CGST Act Sec 132", "3 entities"],
            }
            st.dataframe(pd.DataFrame(outcomes), use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### 📚 Key Learning Points from Case 1")
        learnings = [
            ("Why GST Fraud Fails Benford", "Fabricated invoices show unnatural digit clustering because humans choose amounts mentally, not through real commercial calculations involving rates, quantities, and taxes."),
            ("The Digit-9 Structuring Signal", "When many invoices cluster at ₹90,000–₹99,000 (just below ₹1L reporting threshold), digit-9 over-representation is the telltale sign of deliberate structuring."),
            ("MAD as the Primary Test", f"With 200 invoices, the MAD of {result['mad']:.4f} is highly meaningful. Chi-squared of {result['chi2']:.1f} far exceeds all critical values. Both tests agree: investigate."),
            ("Combine with Other Signals", "Benford alone is a flag, not proof. The actual fraud was confirmed by: e-way bill mismatch, address verification, bank account linkage, and GSTN supplier compliance records."),
            ("Scale of Application", "GSTN can run this test on all taxpayers simultaneously — millions of returns screened in minutes, identifying the highest-risk cases for limited audit resources."),
        ]
        for i, (title, content) in enumerate(learnings, 1):
            st.markdown(f"""
            <div class="mp-card-blue">
              <b style="color:{LIGHT_BLUE};">Learning {i}: {title}</b><br>
              <span style="color:{TXT};">{content}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CASE 2: EXPENSE FRAUD
# ─────────────────────────────────────────────────────────────
elif page == "💼 Case 2: Expense Report Fraud":
    st.markdown("## 💼 Case Study 2: Corporate Expense Report Fraud")

    st.markdown(f"""
    <div class="hero-wrap" style="padding:20px 28px; text-align:left;">
      <span class="badge">Case Study 2</span>
      <span class="badge-red badge">FRAUD DETECTED</span>
      <h3 style="color:{GOLD}; margin:10px 0 6px;">Fabricated Expense Reimbursement Fraud</h3>
      <p style="color:{TXT}; margin:0;">
        <b>Scenario:</b> A pan-India FMCG company processes 6,000 employee expense claims annually.
        The internal audit function deploys Benford's Law analytics on claim amounts across
        five regional sales teams. One regional team's data raises significant red flags.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims Analysed", "6,000")
    col2.metric("Approval Threshold", "₹5,000 (Manager) / ₹25,000 (VP)")
    col3.metric("Suspected Fraudster", "1 Regional Sales Manager")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Background",
        "📊 Benford Analysis",
        "🔍 Per-Employee Analysis",
        "📚 Learning Points",
    ])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""
            <div class="mp-card">
              <b style="color:{GOLD};">Expense Claim Policy</b><br><br>
              The company's expense reimbursement policy:
              <ul style="margin:8px 0; padding-left:20px;">
                <li>Claims ≤ ₹5,000: Self-approved by employee</li>
                <li>Claims ₹5,001–₹25,000: Manager approval</li>
                <li>Claims > ₹25,000: VP Finance approval + supporting docs</li>
                <li>All claims require original receipts</li>
              </ul>
              A fraudulent employee would be motivated to keep claims just below
              the ₹5,000 threshold to avoid any oversight.
            </div>
            <div class="mp-card-red">
              <b style="color:{RED};">The Threshold Gaming Hypothesis</b><br><br>
              If an employee systematically inflates or fabricates claims just below
              the ₹5,000 self-approval threshold, we would expect:
              <ul style="margin:8px 0; padding-left:20px;">
                <li>Spike in claims starting with digit 4 (₹4,000–₹4,999)</li>
                <li>Under-representation of digit 1 and 2 (smaller, legitimate claims)</li>
                <li>Last digits heavily weighted toward 0 and 5 (fabricated round amounts)</li>
                <li>High claim volume from this employee vs. peers</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            np.random.seed(202)
            # Team A: Clean
            team_a_clean = np.concatenate([
                10**np.random.uniform(2, 3.7, 900),
                np.random.exponential(2000, 300),
            ])
            # One fraudster: SM Verma with threshold gaming
            sm_verma = np.concatenate([
                np.random.uniform(4200, 4999, 120),  # Just below 5k
                np.random.choice([4500, 4800, 4999, 4950], 80),  # Round, near limit
                np.random.uniform(100, 3000, 50),    # Some small genuine
            ])
            st.session_state["case2_clean"] = team_a_clean
            st.session_state["case2_fraud"] = sm_verma

            st.markdown("**Claim Count by Employee (Top 10):**")
            emp_df = pd.DataFrame({
                "Employee": ["SM Verma", "R Patel", "A Kumar", "P Singh", "N Mehta",
                             "K Sharma", "S Gupta", "M Iyer", "D Rao", "B Joshi"],
                "Claims": [250, 42, 38, 45, 33, 41, 37, 29, 44, 36],
                "Avg Amount (₹)": ["4,820", "2,340", "1,890", "2,670", "1,540",
                                   "2,100", "1,780", "3,210", "1,990", "2,450"],
            })
            st.dataframe(emp_df, use_container_width=True, hide_index=True)
            st.error("🚨 SM Verma: 250 claims vs. average of 38 for peers — immediate red flag!")

    with tab2:
        col1, col2 = st.columns(2)
        clean_data = st.session_state.get("case2_clean",
            10**np.random.uniform(2, 3.7, 1000))
        fraud_data = st.session_state.get("case2_fraud",
            np.concatenate([np.random.uniform(4200, 4999, 150),
                            np.random.choice([4500, 4800, 4950], 50)]))

        with col1:
            st.markdown("#### ✅ Team Average (Clean Data)")
            r_clean = benford_analysis(clean_data)
            if r_clean:
                show_stats_panel(r_clean)
                st.plotly_chart(
                    chart_first_digit(r_clean, "Team Expenses — Clean"),
                    use_container_width=True
                )

        with col2:
            st.markdown("#### 🚨 SM Verma's Claims (Fraudulent)")
            r_fraud = benford_analysis(fraud_data)
            if r_fraud:
                show_stats_panel(r_fraud)
                st.plotly_chart(
                    chart_first_digit(r_fraud, "SM Verma — Suspicious"),
                    use_container_width=True
                )

        # Side-by-side deviation charts
        col1, col2 = st.columns(2)
        with col1:
            if r_clean:
                st.plotly_chart(chart_deviation(r_clean), use_container_width=True)
        with col2:
            if r_fraud:
                st.plotly_chart(chart_deviation(r_fraud), use_container_width=True)

        # Digit 4 zoom-in
        st.markdown("### 🔍 Digit-4 Deep Dive: Threshold Gaming Evidence")
        col1, col2 = st.columns(2)
        with col1:
            threshold_amounts = fraud_data[(fraud_data >= 4000) & (fraud_data < 5000)]
            fig_th = go.Figure(go.Histogram(
                x=threshold_amounts, nbinsx=40,
                marker_color=RED, opacity=0.8,
            ))
            fig_th.add_vline(x=5000, line_color=GOLD, line_width=2.5,
                             annotation_text="₹5,000 approval limit",
                             annotation_font_color=GOLD)
            fig_th.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="<b>Distribution of Claims ₹4,000–₹5,000</b>",
                           font=dict(color=RED, size=13)),
                xaxis_title="Claim Amount (₹)",
                yaxis_title="Count",
                height=300,
            )
            st.plotly_chart(fig_th, use_container_width=True)

        with col2:
            pct_near_thresh = len(threshold_amounts) / len(fraud_data) * 100
            st.markdown(f"""
            <div class="mp-card-red" style="margin-top:10px;">
              <b style="color:{RED}; font-size:16px;">🚨 Threshold Gaming Confirmed</b><br><br>
              <b style="font-size:22px; color:{GOLD};">{pct_near_thresh:.0f}%</b>
              of SM Verma's claims fall in ₹4,000–₹4,999<br>
              (just below the ₹5,000 self-approval limit)<br><br>
              • Average peer: ~8% of claims near threshold<br>
              • SM Verma: {pct_near_thresh:.0f}% — <b>6× the peer rate</b><br><br>
              Combined with MAD = {r_fraud['mad']:.4f} (non-conforming), this
              constitutes strong grounds for escalation to HR investigation.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 👥 Per-Employee Benford Screening")
        st.info("This demonstrates how to run Benford's Law on each employee's claim history independently — the most powerful application for expense fraud detection.")

        np.random.seed(303)
        employees = {
            "SM Verma":  np.concatenate([np.random.uniform(4200, 4999, 120), np.random.choice([4500, 4800, 4999], 80), np.random.uniform(100, 2000, 50)]),
            "R Patel":   10**np.random.uniform(2, 3.7, 42),
            "A Kumar":   10**np.random.uniform(2, 3.6, 38),
            "P Singh":   10**np.random.uniform(2, 3.8, 45),
            "N Mehta":   10**np.random.uniform(2, 3.5, 33),
            "K Sharma":  np.concatenate([np.random.uniform(24000, 24999, 20), 10**np.random.uniform(2, 3.7, 21)]),
            "S Gupta":   10**np.random.uniform(2, 3.6, 37),
            "M Iyer":    10**np.random.uniform(2, 3.9, 29),
        }
        rows = []
        for name, emp_data in employees.items():
            r = benford_analysis(emp_data)
            if r:
                rows.append({
                    "Employee": name,
                    "Claims": r["n"],
                    "MAD": f"{r['mad']:.4f}",
                    "Chi-Sq": f"{r['chi2']:.1f}",
                    "Verdict": r["verdict"][:25] + "...",
                    "Risk": "🚨 HIGH" if r["mad"] > 0.015 else "⚠️ MED" if r["mad"] > 0.008 else "✅ LOW",
                })
        risk_df = pd.DataFrame(rows)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### 📚 Key Learning Points from Case 2")
        learnings = [
            ("Per-Employee Testing", "The most powerful application is running Benford's Law on each employee's claims individually, not just the aggregate. An individual fraudster can be invisible in aggregate data but stands out clearly in isolation."),
            ("Threshold Gaming = Digit-4 Spike", "When an employee's approval threshold is ₹5,000, expect a spike in digits 4 and possibly 9 (for higher thresholds). The specific digit flagged tells you exactly which threshold is being gamed."),
            ("Volume is a Pre-Screen", "250 claims vs. a peer average of 38 is itself an anomaly. Use transaction count per employee as a first-pass screen before running Benford."),
            ("Last-Digit Test for Round Numbers", "Expense fabricators tend to enter round amounts (₹4,500; ₹4,800; ₹4,950). Last-digit spikes at 0 and 5 confirm fabrication complementing the first-digit test."),
            ("Human Review is Non-Negotiable", f"SM Verma's MAD of {r_fraud['mad']:.4f} is statistically damning, but the final action (termination, legal referral) requires interview, receipt verification, and due process — not just analytics."),
        ]
        for i, (title, content) in enumerate(learnings, 1):
            st.markdown(f"""
            <div class="mp-card-blue">
              <b style="color:{LIGHT_BLUE};">Learning {i}: {title}</b><br>
              <span style="color:{TXT};">{content}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CASE 3: BANK STRUCTURING
# ─────────────────────────────────────────────────────────────
elif page == "🏛️ Case 3: Bank Structuring":
    st.markdown("## 🏛️ Case Study 3: Bank Transaction Structuring (PMLA)")

    st.markdown(f"""
    <div class="hero-wrap" style="padding:20px 28px; text-align:left;">
      <span class="badge">Case Study 3</span>
      <span class="badge-red badge">AML / PMLA</span>
      <h3 style="color:{GOLD}; margin:10px 0 6px;">Cash Deposit Structuring to Evade ₹10 Lakh Reporting</h3>
      <p style="color:{TXT}; margin:0;">
        <b>Scenario:</b> A bank's transaction monitoring system flags a customer who made
        45 cash deposits over 60 days. Individual amounts appear innocuous, but the
        pattern reveals deliberate "structuring" — breaking large sums into smaller
        deposits to stay below the ₹10 lakh PMLA reporting threshold.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deposits", "45 transactions")
    col2.metric("Period", "60 days")
    col3.metric("Total Amount", "≈ ₹43 Lakhs")
    col4.metric("PMLA Threshold", "₹10 Lakhs")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 PMLA Background",
        "📊 Benford Analysis",
        "⏱️ Time-Pattern Analysis",
        "📚 Learning Points",
    ])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""
            <div class="mp-card">
              <b style="color:{GOLD};">PMLA Reporting Obligations (India)</b><br><br>
              Under the Prevention of Money Laundering Act (PMLA) 2002 and RBI guidelines:
              <ul style="margin:8px 0; padding-left:20px;">
                <li>Banks must file <b>Cash Transaction Reports (CTR)</b> for any cash
                    transaction ≥ ₹10 lakh (in a single day)</li>
                <li>Banks must file <b>Suspicious Transaction Reports (STR)</b> to FIU-IND
                    when structuring is suspected</li>
                <li>Structuring (deliberately breaking large amounts) is itself a
                    criminal offence under PMLA Section 3</li>
                <li>Penalty: Rigorous imprisonment 3–7 years + attachment of funds</li>
              </ul>
            </div>
            <div class="mp-card-red">
              <b style="color:{RED};">Structuring Pattern Definition</b><br><br>
              Classic structuring ("smurfing") involves:
              <ul style="margin:8px 0; padding-left:20px;">
                <li>Multiple deposits, each just below the reporting threshold</li>
                <li>Deposits spread across multiple days to avoid daily aggregation</li>
                <li>Sometimes using multiple branches or family members as "smurfs"</li>
                <li>Benford signature: <b>extreme over-representation of digit 9</b>
                    (₹90,000–₹99,999 per deposit when threshold is ₹1L)</li>
                <li>Or digit 4 and 5 spikes when gaming ₹50,000 sub-thresholds</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Structuring data
            np.random.seed(404)
            structuring_deposits = np.concatenate([
                np.random.uniform(920000, 998000, 30),   # Digit-9: just below ₹10L
                np.random.uniform(850000, 899000, 10),   # Digit-8
                np.random.uniform(970000, 999000, 5),    # More digit-9
            ])
            st.session_state["case3_data"] = structuring_deposits

            st.markdown("**Deposit History (₹):**")
            dates_dep = pd.date_range("2024-01-02", periods=len(structuring_deposits), freq="D")
            dep_df = pd.DataFrame({
                "Date": dates_dep[:15].strftime("%d-%b-%Y"),
                "Amount (₹)": [f"₹{x:,.0f}" for x in structuring_deposits[:15]],
                "< ₹10L?": ["✅ Yes" for _ in range(15)],
            })
            st.dataframe(dep_df, use_container_width=True, hide_index=True)
            st.error("⚠️ Every single deposit is below ₹10 lakh — none trigger CTR individually!")

    with tab2:
        deposits = st.session_state.get("case3_data",
            np.random.uniform(920000, 998000, 45))
        result = benford_analysis(deposits)

        show_stats_panel(result)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                chart_first_digit(result, "Cash Deposit Structuring Analysis"),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(chart_deviation(result), use_container_width=True)

        show_digit_table(result)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_log_histogram(deposits), use_container_width=True)
        with col2:
            # Amount distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=deposits/100000,
                nbinsx=30,
                marker_color=RED,
                opacity=0.8,
                name="Deposits",
            ))
            fig_dist.add_vline(x=10, line_color=GOLD, line_width=2.5,
                               annotation_text="₹10L PMLA threshold",
                               annotation_font_color=GOLD)
            fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="<b>All Deposits Cluster Just Below ₹10 Lakh</b>",
                           font=dict(color=RED, size=13)),
                xaxis_title="Deposit Amount (₹ Lakhs)",
                yaxis_title="Frequency",
                height=320,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown(f"""
        <div class="verdict-bad">
          🚨 STRUCTURING CONFIRMED — DIGIT-9 FREQUENCY: {result['obs_p'][9]*100:.1f}%
          (Expected: 4.6%) | MAD = {result['mad']:.4f} | File STR with FIU-IND
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### ⏱️ Time-Pattern and Velocity Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Simulate time series of deposits
            np.random.seed(505)
            n_deps = len(st.session_state.get("case3_data", [45]))
            dates_series = pd.date_range("2024-01-02", periods=45, freq="D")
            amounts_series = st.session_state.get("case3_data",
                                                   np.random.uniform(920000, 998000, 45))

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=dates_series,
                y=amounts_series/100000,
                mode="markers+lines",
                marker=dict(color=RED, size=8),
                line=dict(color=MID_BLUE, width=1.5),
                name="Deposit Amount",
            ))
            fig_ts.add_hline(y=10, line_color=GOLD, line_width=2,
                             line_dash="dash",
                             annotation_text="₹10L threshold",
                             annotation_font_color=GOLD)
            fig_ts.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="<b>Daily Cash Deposits Over 60 Days</b>",
                           font=dict(color=GOLD, size=13)),
                xaxis_title="Date",
                yaxis_title="Amount (₹ Lakhs)",
                height=320,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with col2:
            # Rolling cumulative
            cumulative = np.cumsum(amounts_series)
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=dates_series,
                y=cumulative/100000,
                fill="tozeroy",
                fillcolor=f"{MID_BLUE}44",
                line=dict(color=DARK_BLUE, width=2),
                name="Cumulative",
            ))
            fig_cum.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="<b>Cumulative Deposits Over 60 Days</b>",
                           font=dict(color=GOLD, size=13)),
                xaxis_title="Date",
                yaxis_title="Cumulative (₹ Lakhs)",
                height=320,
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        total = sum(amounts_series)
        st.markdown(f"""
        <div class="mp-card-red">
          <b style="color:{RED};">Aggregate Picture Reveals the Scheme</b><br><br>
          • <b>45 deposits × average ₹{total/len(amounts_series)/100000:.1f} lakhs
            = Total ₹{total/100000:.1f} lakhs</b> deposited in 60 days<br>
          • <b>0 individual deposits</b> triggered a CTR (all below ₹10L)<br>
          • <b>Aggregate</b> clearly indicates a large undisclosed income source<br>
          • <b>Velocity:</b> 45 cash deposits in 60 days = 0.75 deposits/day
            (extreme for a retail account — peer average: 2-3 per month)<br><br>
          <b>Under PMLA, the bank's obligation is to file an STR even when no single
          transaction crosses the threshold, if the overall pattern is suspicious.</b>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 📚 Key Learning Points from Case 3")
        learnings = [
            ("Structuring = Digit-9 Signature", "When the PMLA threshold is ₹10 lakh, structuring creates an extreme digit-9 spike (₹90,000–₹9,99,999). The specific digit-spike reveals exactly which threshold is being gamed."),
            ("Aggregate Matters as Much as Individual", "No single transaction triggered a CTR. But the aggregate over 60 days is ₹43 lakhs. Benford analysis on the time-aggregated transaction history reveals the scheme."),
            ("Benford + Velocity = Powerful Combination", "Combining Benford non-conformity (which digit) with velocity analysis (how often) gives investigators a complete picture: what amounts and at what pace."),
            ("Legal Obligation: STR Filing", "Under PMLA and RBI Master Directions, banks must file STRs when structuring is suspected — regardless of individual transaction size. Benford analysis provides the analytical basis for the STR."),
            ("FIU-IND Integration", "In practice, Indian banks' transaction monitoring systems are integrated with FIU-IND (Financial Intelligence Unit). STRs filed based on Benford-triggered alerts are reviewed for prosecution under PMLA Section 3."),
        ]
        for i, (title, content) in enumerate(learnings, 1):
            st.markdown(f"""
            <div class="mp-card-blue">
              <b style="color:{LIGHT_BLUE};">Learning {i}: {title}</b><br>
              <span style="color:{TXT};">{content}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ML ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────
elif page == "🤖 ML Anomaly Detection":
    st.markdown("## 🤖 Machine Learning Anomaly Detection")

    tab1, tab2, tab3 = st.tabs([
        "🌲 Isolation Forest",
        "📐 Multi-Method Framework",
        "📚 ML Concepts",
    ])

    with tab1:
        st.markdown("### 🌲 Isolation Forest for Transaction Anomaly Detection")
        st.markdown(f"""
        <div class="mp-card-blue">
          <b style="color:{LIGHT_BLUE};">How Isolation Forest Works</b><br><br>
          The algorithm builds random decision trees. <b>Anomalous transactions are
          "easy to isolate"</b> — they require very few random splits to separate
          from the rest. Normal transactions require many splits because they are
          densely packed together.<br><br>
          Anomaly score ≈ 1: Clearly anomalous | ≈ 0.5: Normal | ≪ 0.5: Very normal
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Configure the Model:**")
            n_total = st.slider("Total transactions:", 1000, 5000, 3000, 100)
            contamination = st.slider("Expected fraud rate (%):", 1, 10, 3) / 100
            n_estimators = st.selectbox("Number of trees:", [100, 200, 300], index=1)

            st.markdown("**Fraud Injection Settings:**")
            inc_structuring = st.checkbox("Include structuring fraud", True)
            inc_velocity    = st.checkbox("Include high-velocity fraud", True)
            inc_large       = st.checkbox("Include large amount fraud", True)

            run_ml = st.button("🚀 Run Isolation Forest", type="primary")

        with col2:
            if run_ml:
                np.random.seed(42)
                n_clean = int(n_total * 0.95)
                n_fraud = n_total - n_clean

                clean_amounts = np.random.lognormal(9, 1.5, n_clean)
                clean_hours   = np.random.randint(8, 22, n_clean)
                clean_velocity= np.random.poisson(3, n_clean)
                clean_days    = np.random.exponential(2, n_clean)

                fraud_amounts, fraud_hours, fraud_velocity, fraud_days = [], [], [], []

                if inc_structuring:
                    nf = n_fraud // 3
                    fraud_amounts.extend(np.random.uniform(950000, 999000, nf))
                    fraud_hours.extend(np.random.randint(8, 22, nf))
                    fraud_velocity.extend(np.random.poisson(3, nf))
                    fraud_days.extend(np.random.exponential(2, nf))

                if inc_velocity:
                    nf = n_fraud // 3
                    fraud_amounts.extend(np.random.lognormal(9, 1.5, nf))
                    fraud_hours.extend(np.random.randint(22, 24, nf))
                    fraud_velocity.extend(np.random.poisson(30, nf))
                    fraud_days.extend(np.random.uniform(0, 0.1, nf))

                if inc_large:
                    nf = n_fraud - len(fraud_amounts)
                    if nf > 0:
                        fraud_amounts.extend(np.random.uniform(5000000, 9999999, nf))
                        fraud_hours.extend(np.random.randint(8, 22, nf))
                        fraud_velocity.extend(np.random.poisson(5, nf))
                        fraud_days.extend(np.random.exponential(1, nf))

                all_amounts   = np.concatenate([clean_amounts, fraud_amounts])
                all_hours     = np.concatenate([clean_hours, fraud_hours])
                all_velocity  = np.concatenate([clean_velocity, fraud_velocity])
                all_days      = np.concatenate([clean_days, fraud_days])
                true_labels   = np.concatenate([
                    np.zeros(n_clean), np.ones(len(fraud_amounts))
                ])

                # Benford feature
                def get_fd(x):
                    try:
                        return int(str(abs(int(x)))[0])
                    except:
                        return 1
                first_digits  = np.array([get_fd(x) for x in all_amounts])
                benford_exp   = np.array([BENFORD.get(d, 0.1) for d in first_digits])

                df_ml = pd.DataFrame({
                    "log_amount": np.log10(np.maximum(all_amounts, 1)),
                    "hour": all_hours,
                    "velocity": all_velocity,
                    "days_since_last": all_days,
                    "benford_expected": benford_exp,
                })

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_ml)

                iso = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42,
                )
                preds = iso.fit_predict(X_scaled)
                scores = iso.score_samples(X_scaled)

                df_ml["predicted_fraud"] = (preds == -1).astype(int)
                df_ml["anomaly_score"]   = scores
                df_ml["true_fraud"]      = true_labels.astype(int)
                df_ml["amount"]          = all_amounts

                tp = int(((df_ml["predicted_fraud"]==1) & (df_ml["true_fraud"]==1)).sum())
                fp = int(((df_ml["predicted_fraud"]==1) & (df_ml["true_fraud"]==0)).sum())
                fn = int(((df_ml["predicted_fraud"]==0) & (df_ml["true_fraud"]==1)).sum())
                tn = int(((df_ml["predicted_fraud"]==0) & (df_ml["true_fraud"]==0)).sum())

                precision = tp / (tp + fp) if (tp+fp) > 0 else 0
                recall    = tp / (tp + fn) if (tp+fn) > 0 else 0
                f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

                c1m, c2m, c3m, c4m = st.columns(4)
                c1m.metric("Precision", f"{precision*100:.1f}%")
                c2m.metric("Recall (Detection Rate)", f"{recall*100:.1f}%")
                c3m.metric("F1-Score", f"{f1:.3f}")
                c4m.metric("Flagged", f"{tp+fp:,}")

                # Scatter plot
                normal_df = df_ml[df_ml["predicted_fraud"]==0]
                fraud_df  = df_ml[df_ml["predicted_fraud"]==1]

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=normal_df["log_amount"],
                    y=normal_df["velocity"],
                    mode="markers",
                    marker=dict(color=MID_BLUE, size=4, opacity=0.4),
                    name="Normal",
                ))
                fig_sc.add_trace(go.Scatter(
                    x=fraud_df["log_amount"],
                    y=fraud_df["velocity"],
                    mode="markers",
                    marker=dict(color=RED, size=8, symbol="x", opacity=0.8),
                    name="Flagged Anomaly",
                ))
                fig_sc.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text="<b>Isolation Forest: Amount vs Velocity</b>",
                               font=dict(color=GOLD, size=13)),
                    xaxis_title="log₁₀(Amount)",
                    yaxis_title="Transaction Velocity (per day)",
                    height=380,
                    legend=dict(font=dict(color=TXT)),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

                # Anomaly score distribution
                fig_score = go.Figure()
                fig_score.add_trace(go.Histogram(
                    x=df_ml[df_ml["predicted_fraud"]==0]["anomaly_score"],
                    name="Normal", marker_color=DARK_BLUE, opacity=0.75,
                    nbinsx=50, histnorm="density",
                ))
                fig_score.add_trace(go.Histogram(
                    x=df_ml[df_ml["predicted_fraud"]==1]["anomaly_score"],
                    name="Anomaly", marker_color=RED, opacity=0.85,
                    nbinsx=30, histnorm="density",
                ))
                fig_score.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text="<b>Anomaly Score Distribution</b>",
                               font=dict(color=GOLD, size=13)),
                    xaxis_title="Isolation Forest Score",
                    yaxis_title="Density",
                    barmode="overlay",
                    height=300,
                    legend=dict(font=dict(color=TXT)),
                )
                st.plotly_chart(fig_score, use_container_width=True)

    with tab2:
        st.markdown("### 📐 Multi-Method Fraud Detection Framework")
        st.markdown(f"""
        <div class="mp-card">
          <b style="color:{GOLD};">Ensemble Anomaly Scoring</b><br><br>
          No single method is universally best. The most robust systems combine
          multiple signals into a <b>composite fraud risk score</b>:
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">ENSEMBLE SCORE FORMULA</div>
              <div style="color:{GOLD}; font-size:16px; font-family:Georgia,serif;">
                Score = w₁·S_Benford + w₂·S_IsoForest + w₃·S_Rules + w₄·S_Network
              </div>
              <div style="color:{LIGHT_BLUE}; font-size:11px; margin-top:6px;">
                Each S normalised to [0,1] | Σwᵢ = 1
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            weights_df = pd.DataFrame({
                "Component": ["Rule-Based Detection", "Benford's Law", "Isolation Forest", "Network / Velocity"],
                "Weight": ["25%", "20%", "30%", "25%"],
                "Detects Best": ["Known patterns", "Fabricated amounts", "Multivariate anomalies", "Collaborative fraud"],
            })
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

        # 5-layer framework
        layers = [
            ("Layer 1: Rule-Based", DARK_BLUE, "Transaction limits • Threshold proximity • Duplicate checks • Off-hours alerts"),
            ("Layer 2: Statistical (Benford)", MID_BLUE, "First-digit test • MAD • Chi-squared • Z-score per digit • Summation test"),
            ("Layer 3: Machine Learning", "#6a0080", "Isolation Forest • Autoencoder • DBSCAN • Local Outlier Factor"),
            ("Layer 4: Network Analysis", "#008060", "Graph analytics • Entity linkages • Peer comparison • Velocity analysis"),
            ("Layer 5: Human Investigation", "#805000", "Prioritised case queue • Interview • Physical verification • Legal action"),
        ]
        for label, color, desc in layers:
            st.markdown(f"""
            <div style="background:{CARD_BG}; border-left:5px solid {color};
                border-radius:6px; padding:12px 16px; margin:6px 0;">
              <b style="color:{color};">{label}</b>
              <span style="color:{MUTED}; font-size:12px; margin-left:12px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 📚 ML Concepts Explained Simply")
        concepts = [
            ("Supervised vs Unsupervised", "Fraud detection faces the challenge of very few labeled examples. Isolation Forest is **unsupervised** — it learns what 'normal' looks like without needing fraud labels. This is critical in finance where fraud is rare and often novel."),
            ("The Curse of Class Imbalance", "In real fraud data, genuine fraud is typically 0.1–3% of transactions. A naive model that labels everything 'normal' achieves 99% accuracy but catches no fraud. Use Precision, Recall, and F1 — not accuracy."),
            ("Precision vs Recall Trade-off", "High Precision = fewer false positives (fewer innocent customers wrongly flagged). High Recall = fewer missed frauds. In fraud detection, Recall is usually more important — it's worse to miss fraud than to over-investigate."),
            ("Feature Engineering for Fraud", "The most important ML skill in fraud detection is creating meaningful features: log-transform amounts, compute velocity (txns per day), time-since-last (recency), Benford's expected probability as a feature, rolling averages."),
            ("Model Drift", "Fraud patterns evolve. A model trained on 2022 data may miss novel 2025 fraud patterns. Re-train quarterly with new confirmed fraud cases. Track model performance (Recall) over time — declining Recall signals drift."),
        ]
        for title, content in concepts:
            with st.expander(f"💡 {title}", expanded=False):
                st.markdown(f"""
                <div class="mp-card-blue">
                  <span style="color:{TXT};">{content}</span>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# QUIZ
# ─────────────────────────────────────────────────────────────
elif page == "❓ Quiz & Assessment":
    st.markdown("## ❓ Knowledge Assessment — Benford's Law & Fraud Analytics")

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False

    questions = [
        {
            "q": "1. According to Benford's Law, what is the expected probability of a number beginning with the digit 1?",
            "options": ["11.1% (uniform)", "30.1%", "17.6%", "25.0%"],
            "answer": 1,
            "explain": "P(1) = log₁₀(1 + 1/1) = log₁₀(2) ≈ 30.1%. The digit 1 appears as the leading digit in roughly 1 in 3 naturally occurring numbers.",
        },
        {
            "q": "2. In a dataset of 1,000 expense claims, you compute MAD = 0.022. What is the correct interpretation?",
            "options": [
                "Close conformity — no issue",
                "Acceptable conformity — monitor",
                "Non-conforming — investigate immediately",
                "Too few observations to conclude",
            ],
            "answer": 2,
            "explain": "MAD > 0.015 indicates non-conformity with Benford's Law. This dataset should be flagged for investigation (Nigrini, 2012).",
        },
        {
            "q": "3. A bank customer makes 30 deposits of exactly ₹9,50,000 each over 90 days (PMLA threshold: ₹10 lakh). Which fraud type does this indicate?",
            "options": [
                "Tax evasion through round numbers",
                "Structuring (smurfing) under PMLA threshold",
                "Financial statement manipulation",
                "Expense reimbursement fraud",
            ],
            "answer": 1,
            "explain": "This is classic structuring (smurfing) — deliberately breaking a large sum into deposits below the ₹10L Cash Transaction Report threshold to avoid PMLA reporting obligations.",
        },
        {
            "q": "4. Which of these datasets does NOT follow Benford's Law?",
            "options": [
                "NSE daily stock trading volumes",
                "GST invoice amounts filed by traders",
                "Employee ID numbers (sequential)",
                "Financial statement line items (CMIE Prowess)",
            ],
            "answer": 2,
            "explain": "Sequential employee IDs are assigned uniformly (1, 2, 3...) — not generated by natural multiplicative processes. They do not follow Benford's Law.",
        },
        {
            "q": "5. When chi-squared rejects Benford's Law at p=0.001 but MAD=0.008 (acceptable conformity), what should you conclude?",
            "options": [
                "Fraud confirmed — both tests agree",
                "Chi-squared wins — investigate immediately",
                "MAD is more reliable here; the deviation may be statistically significant but practically small",
                "Discard the chi-squared test completely",
            ],
            "answer": 2,
            "explain": "With very large samples, chi-squared becomes overly sensitive and rejects Benford even for tiny, practically insignificant deviations. MAD is the preferred metric for large financial datasets.",
        },
        {
            "q": "6. You run Benford's Law on 200 sales invoices and digit 4 has a Z-score of 5.2 with your approval threshold at ₹5,000. What fraud hypothesis fits best?",
            "options": [
                "Invoices are priced with psychological anchoring (₹4,999)",
                "Threshold gaming — amounts systematically inflated just below ₹5,000",
                "Rounding errors in the accounting system",
                "The dataset is too small for Benford's Law",
            ],
            "answer": 1,
            "explain": "Digit-4 over-representation (amounts ₹4,000–₹4,999) when the approval threshold is ₹5,000 is the classic signature of threshold gaming — deliberately keeping claims just below the manager review limit.",
        },
        {
            "q": "7. A sophisticated fraudster who knows Benford's Law deliberately fabricates 500 invoices that conform to the expected first-digit distribution. What does this imply?",
            "options": [
                "Benford's Law is useless for fraud detection",
                "Benford should always be supplemented by other tests (second digit, summation, last digit, ML)",
                "The fraudster cannot be caught",
                "Use a larger sample size to detect the fraud",
            ],
            "answer": 1,
            "explain": "Benford's Law is one tool, not a silver bullet. A knowledgeable fraudster can defeat first-digit conformity but is unlikely to simultaneously satisfy second-digit, summation, last-digit, and ML-based anomaly tests.",
        },
        {
            "q": "8. Which action is required under Indian law when a bank's analytics system detects PMLA structuring?",
            "options": [
                "Freeze the account immediately without notice",
                "Inform the customer and give 30 days to explain",
                "File a Suspicious Transaction Report (STR) with FIU-IND",
                "Wait for ₹10L threshold to be crossed before acting",
            ],
            "answer": 2,
            "explain": "Under PMLA and RBI Master Directions, banks must file an STR with FIU-IND when structuring is suspected — regardless of individual transaction size. The obligation exists even when no single transaction crosses the CTR threshold.",
        },
    ]

    if not st.session_state.quiz_submitted:
        st.markdown(f"""
        <div class="mp-card">
          <b style="color:{GOLD};">📋 Instructions</b><br>
          Answer all 8 questions, then click Submit to see your score and detailed explanations.
          Each question carries 1 mark.
        </div>
        """, unsafe_allow_html=True)

        for i, q_data in enumerate(questions):
            sel = st.radio(
                q_data["q"],
                q_data["options"],
                key=f"q{i}",
                index=None,
            )
            if sel:
                st.session_state.quiz_answers[i] = q_data["options"].index(sel)

        if st.button("✅ Submit Assessment", type="primary"):
            if len(st.session_state.quiz_answers) < len(questions):
                st.warning(f"Please answer all questions. ({len(st.session_state.quiz_answers)}/{len(questions)} answered)")
            else:
                score = sum(
                    1 for i, q in enumerate(questions)
                    if st.session_state.quiz_answers.get(i) == q["answer"]
                )
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True
                st.rerun()

    else:
        score = st.session_state.quiz_score
        pct   = score / len(questions) * 100

        if pct >= 75:
            grade, css = f"Excellent! {score}/{len(questions)} ({pct:.0f}%)", "verdict-ok"
        elif pct >= 50:
            grade, css = f"Good effort! {score}/{len(questions)} ({pct:.0f}%)", "verdict-warn"
        else:
            grade, css = f"Needs Review: {score}/{len(questions)} ({pct:.0f}%)", "verdict-bad"

        st.markdown(f'<div class="{css}">🎓 Your Score: {grade}</div>', unsafe_allow_html=True)
        st.progress(score / len(questions))

        st.markdown("### 📖 Detailed Explanations")
        for i, q_data in enumerate(questions):
            user_ans = st.session_state.quiz_answers.get(i, -1)
            correct  = q_data["answer"]
            correct_flag = user_ans == correct

            with st.expander(
                f"{'✅' if correct_flag else '❌'} Q{i+1}: {q_data['q'][:70]}...",
                expanded=not correct_flag,
            ):
                st.markdown(f"**Your answer:** {q_data['options'][user_ans] if user_ans >= 0 else 'Not answered'}")
                st.markdown(f"**Correct answer:** {q_data['options'][correct]}")
                if correct_flag:
                    st.success(f"✅ Correct! {q_data['explain']}")
                else:
                    st.error(f"❌ Explanation: {q_data['explain']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Quiz"):
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()
        with col2:
            st.markdown(f"""
            <div class="mp-card">
              <b style="color:{GOLD};">📚 Continue Learning</b><br>
              Visit <a href="https://themountainpathacademy.com">themountainpathacademy.com</a>
              for the full LaTeX notes, Python notebooks, and extended case studies.
            </div>
            """, unsafe_allow_html=True)

    # Resources footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{MUTED}; font-size:12px; padding:10px;">
      <b style="color:{GOLD};">The Mountain Path — World of Finance</b><br>
      Prof. V. Ravichandran |
      <a href="https://themountainpathacademy.com">themountainpathacademy.com</a> |
      <a href="https://www.linkedin.com/in/trichyravis">LinkedIn</a> |
      <a href="https://github.com/trichyravis">GitHub</a><br>
      <span style="font-size:10px; color:{MUTED}22;">
        Reference: Nigrini (2012) Benford's Law | ACFE 2024 Report to the Nations
      </span>
    </div>
    """, unsafe_allow_html=True)
