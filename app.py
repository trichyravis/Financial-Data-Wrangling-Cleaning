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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import zipfile, io
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
# DESIGN TOKENS
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

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{
    background: {BG_GRAD} !important;
    color: {TXT} !important;
    font-family: 'Segoe UI', Arial, sans-serif;
  }}
  .block-container {{ padding-top: 1.5rem; }}

  section[data-testid="stSidebar"] {{
    background: #0a1628 !important;
    border-right: 3px solid {GOLD} !important;
  }}
  section[data-testid="stSidebar"] * {{ color: #ffffff !important; }}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] .stRadio p,
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
    color: #ffffff !important; font-size: 13px !important;
  }}
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
    background: {DARK_BLUE}aa !important; border-radius: 6px; color: {GOLD} !important;
  }}
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] h4 {{ color: {GOLD} !important; }}
  section[data-testid="stSidebar"] a {{ color: {GOLD} !important; text-decoration: none; }}
  section[data-testid="stSidebar"] a:hover {{ color: {LIGHT_BLUE} !important; }}
  section[data-testid="stSidebar"] hr {{ border-color: {GOLD}55 !important; }}
  section[data-testid="stSidebar"] .stRadio > label {{
    color: {GOLD} !important; font-weight: 700 !important;
    font-size: 13px !important; letter-spacing: 0.5px; text-transform: uppercase;
  }}

  h1, h2, h3, h4 {{ color: {GOLD} !important; }}
  p, li, span {{ color: {TXT}; }}

  [data-testid="metric-container"] {{
    background: #0d1b2e !important; border: 1px solid {GOLD}55 !important;
    border-radius: 10px !important; padding: 14px !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricLabel"] p {{
    color: {LIGHT_BLUE} !important; font-size: 12px !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {GOLD} !important; font-size: 26px !important; font-weight: 800 !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    color: {MUTED} !important; font-size: 11px !important;
  }}

  .stTabs [data-baseweb="tab-list"] {{
    background: #0d1b2e !important; border-radius: 8px 8px 0 0;
    gap: 3px; padding: 4px 4px 0; border-bottom: 2px solid {GOLD}33;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {LIGHT_BLUE} !important; font-weight: 600 !important;
    font-size: 13px !important; border-radius: 6px 6px 0 0; padding: 8px 16px;
    background: #152035 !important; border: 1px solid {GOLD}22 !important;
    border-bottom: none !important; transition: all 0.2s;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color: {GOLD} !important; background: {DARK_BLUE} !important; }}
  .stTabs [aria-selected="true"] {{
    background: {DARK_BLUE} !important; color: {GOLD} !important;
    border-bottom: 3px solid {GOLD} !important; font-weight: 700 !important;
  }}

  .stAlert {{ border-radius: 8px !important; }}
  div[data-testid="stInfo"] {{
    background: #0d2240 !important; border-left: 4px solid {LIGHT_BLUE} !important; color: {TXT} !important;
  }}
  div[data-testid="stSuccess"] {{
    background: #0a2a14 !important; border-left: 4px solid {GREEN} !important; color: #c3f0ca !important;
  }}
  div[data-testid="stWarning"] {{
    background: #2a1f00 !important; border-left: 4px solid {GOLD} !important; color: #ffe9a0 !important;
  }}
  div[data-testid="stError"] {{
    background: #2a0a0a !important; border-left: 4px solid {RED} !important; color: #ffb3b3 !important;
  }}

  .stTextArea textarea {{
    background: #0d1b2e !important; color: {TXT} !important;
    border: 1px solid {GOLD}44 !important; border-radius: 6px !important;
    font-family: 'Courier New', monospace;
  }}
  .stSelectbox [data-baseweb="select"] div,
  .stSelectbox [data-baseweb="select"] span {{
    background: #0d1b2e !important; color: {TXT} !important; border-color: {GOLD}44 !important;
  }}
  .main .stRadio label p, .main .stRadio label span {{ color: {TXT} !important; font-size: 13px !important; }}
  .stCheckbox label p, .stCheckbox label span {{ color: {TXT} !important; }}
  .stSlider [data-baseweb="slider"] {{ color: {GOLD} !important; }}

  [data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
  .stDataFrame thead tr th {{
    background: {DARK_BLUE} !important; color: {GOLD} !important; font-weight: 700 !important;
  }}
  .stDataFrame tbody tr td {{ color: {TXT} !important; background: #0d1b2e !important; }}
  .stDataFrame tbody tr:nth-child(even) td {{ background: #112240 !important; }}

  [data-testid="stFileUploader"] {{
    background: #0d1b2e !important; border: 2px dashed {GOLD}55 !important; border-radius: 8px !important;
  }}
  [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p {{ color: {LIGHT_BLUE} !important; }}

  /* Expander — modern + legacy selectors */
  .streamlit-expanderHeader {{
    background: #0d1b2e !important; color: {GOLD} !important;
    border-radius: 6px !important; font-weight: 600 !important;
  }}
  .streamlit-expanderContent {{
    background: #0a1628 !important; border: 1px solid {GOLD}22 !important;
  }}
  [data-testid="stExpander"] {{
    background: #0d1b2e !important; border: 1px solid {GOLD}33 !important; border-radius: 8px !important;
  }}
  [data-testid="stExpander"] summary {{
    background: #0d1b2e !important; color: {LIGHT_BLUE} !important;
    border-radius: 8px !important; padding: 10px 14px !important;
  }}
  [data-testid="stExpander"] summary:hover {{ background: {DARK_BLUE} !important; color: {GOLD} !important; }}
  [data-testid="stExpander"][open] summary {{
    background: {DARK_BLUE} !important; color: {GOLD} !important;
    border-bottom: 1px solid {GOLD}44 !important; border-radius: 8px 8px 0 0 !important;
  }}
  [data-testid="stExpander"] summary *,
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary span {{ color: {LIGHT_BLUE} !important; font-weight: 600 !important; }}
  [data-testid="stExpander"] summary:hover *,
  [data-testid="stExpander"][open] summary * {{ color: {GOLD} !important; }}
  [data-testid="stExpander"] summary svg {{ fill: {LIGHT_BLUE} !important; }}
  [data-testid="stExpander"] summary:hover svg,
  [data-testid="stExpander"][open] summary svg {{ fill: {GOLD} !important; }}

  .stButton > button {{
    background: {DARK_BLUE} !important; color: {GOLD} !important;
    border: 2px solid {GOLD} !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 10px 24px !important; transition: all 0.2s !important;
  }}
  .stButton > button:hover {{ background: {GOLD} !important; color: {DARK_BLUE} !important; }}
  .stButton > button[kind="primary"] {{ background: {GOLD} !important; color: {DARK_BLUE} !important; }}
  .stButton > button[kind="primary"]:hover {{ background: #e6c200 !important; }}

  /* Download buttons — match app design */
  [data-testid="stDownloadButton"] > button {{
    background: {DARK_BLUE} !important; color: {GOLD} !important;
    border: 2px solid {GOLD} !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 10px 24px !important; width: 100% !important; transition: all 0.2s !important;
  }}
  [data-testid="stDownloadButton"] > button:hover {{
    background: {GOLD} !important; color: {DARK_BLUE} !important;
  }}
  [data-testid="stDownloadButton"] > button p,
  [data-testid="stDownloadButton"] > button span,
  [data-testid="stDownloadButton"] > button div {{ color: {GOLD} !important; font-weight: 700 !important; }}
  [data-testid="stDownloadButton"] > button:hover p,
  [data-testid="stDownloadButton"] > button:hover span,
  [data-testid="stDownloadButton"] > button:hover div {{ color: {DARK_BLUE} !important; }}

  .stProgress > div > div > div > div {{ background: {GOLD} !important; }}
  .stProgress > div > div > div {{ background: #0d1b2e !important; }}

  .mp-card {{
    background: #0d1b2e; border: 1px solid {GOLD}44;
    border-left: 4px solid {GOLD}; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 14px; color: {TXT};
  }}
  .mp-card-red {{
    background: #1a0a0a; border: 1px solid {RED}66;
    border-left: 4px solid {RED}; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 14px; color: {TXT};
  }}
  .mp-card-green {{
    background: #0a1a0a; border: 1px solid {GREEN}66;
    border-left: 4px solid {GREEN}; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 14px; color: {TXT};
  }}
  .mp-card-blue {{
    background: #0a1428; border: 1px solid {LIGHT_BLUE}66;
    border-left: 4px solid {LIGHT_BLUE}; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 14px; color: {TXT};
  }}
  .hero-wrap {{
    background: linear-gradient(135deg,{DARK_BLUE},{MID_BLUE});
    border: 2px solid {GOLD}; border-radius: 14px;
    padding: 28px 34px; text-align: center; margin-bottom: 22px;
  }}
  .badge {{
    display: inline-block; background: {GOLD}; color: {DARK_BLUE};
    font-weight: 700; font-size: 11px; padding: 3px 10px;
    border-radius: 20px; margin: 2px; user-select: none;
  }}
  .badge-red {{ background: {RED}; color: #ffffff; }}
  .badge-green {{ background: {GREEN}; color: #ffffff; }}
  .formula-box {{
    background: linear-gradient(135deg,#001a40,{DARK_BLUE});
    border: 2px solid {GOLD}; border-radius: 10px;
    padding: 18px 24px; text-align: center; margin: 14px 0; user-select: none;
  }}
  .verdict-ok {{
    background: linear-gradient(90deg,#0d2e0d,#0a1628); border-left: 5px solid {GREEN};
    border-radius: 8px; padding: 14px 20px; color: #c3f0ca; font-weight: 700;
    font-size: 15px; margin: 10px 0;
  }}
  .verdict-warn {{
    background: linear-gradient(90deg,#2a1f00,#0a1628); border-left: 5px solid {GOLD};
    border-radius: 8px; padding: 14px 20px; color: {GOLD}; font-weight: 700;
    font-size: 15px; margin: 10px 0;
  }}
  .verdict-bad {{
    background: linear-gradient(90deg,#2a0000,#0a1628); border-left: 5px solid {RED};
    border-radius: 8px; padding: 14px 20px; color: #ffb3b3; font-weight: 700;
    font-size: 15px; margin: 10px 0;
  }}
  a {{ color: {GOLD} !important; text-decoration: none; }}
  a:hover {{ color: {LIGHT_BLUE} !important; text-decoration: underline; }}
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
    if n == 0: return None
    digits = extract_digits(data)
    counts = {d: digits.count(d) for d in range(1, 10)}
    obs_p  = {d: counts[d] / n for d in range(1, 10)}
    exp_p  = BENFORD.copy()
    expected = {d: exp_p[d] * n for d in range(1, 10)}
    chi2 = sum((counts[d] - expected[d])**2 / expected[d] for d in range(1, 10))
    p_chi2 = 1 - stats.chi2.cdf(chi2, df=8)
    mad = sum(abs(obs_p[d] - exp_p[d]) for d in range(1, 10)) / 9
    z = {}
    for d in range(1, 10):
        p_b = exp_p[d]
        se = np.sqrt(p_b * (1 - p_b) / n)
        z[d] = (abs(obs_p[d] - p_b) - 1/(2*n)) / se if se > 0 else 0
    cdf_obs = np.cumsum([obs_p[d] for d in range(1, 10)])
    cdf_ben = np.cumsum([exp_p[d] for d in range(1, 10)])
    ks_stat = float(np.max(np.abs(cdf_obs - cdf_ben)))
    if mad < 0.006:   verdict, level = "CLOSE CONFORMITY — No significant anomaly", "ok"
    elif mad < 0.012: verdict, level = "ACCEPTABLE CONFORMITY — Monitor", "warn"
    elif mad < 0.015: verdict, level = "MARGINAL CONFORMITY — Review Recommended", "warn"
    else:             verdict, level = "NON-CONFORMING — INVESTIGATE IMMEDIATELY", "bad"
    return dict(n=n, counts=counts, obs_p=obs_p, exp_p=exp_p, expected=expected,
                chi2=chi2, p_chi2=p_chi2, mad=mad, z=z, ks=ks_stat,
                verdict=verdict, level=level, digits=digits)

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
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG,
    font=dict(color=TXT, family="Segoe UI, Arial"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def chart_first_digit(result, title="Benford Analysis"):
    digits = list(range(1, 10))
    obs = [result["obs_p"][d]*100 for d in digits]
    ben = [result["exp_p"][d]*100 for d in digits]
    z_vals = [result["z"][d] for d in digits]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["First-Digit Frequency vs Benford's Law",
                        "Z-Score per Digit (Statistical Significance)"],
        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=[str(d) for d in digits], y=obs, name="Observed",
        marker_color=DARK_BLUE, marker_line=dict(color=GOLD, width=1.2),
        hovertemplate="Digit %{x}<br>Observed: %{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=[str(d) for d in digits], y=ben, name="Benford's Law",
        marker_color=GOLD, marker_opacity=0.85,
        hovertemplate="Digit %{x}<br>Expected: %{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[str(d) for d in digits], y=ben,
        mode="lines+markers", name="Benford Curve",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(symbol="circle-open", size=7, color="white"), showlegend=False), row=1, col=1)
    z_colors = [RED if z > 2.576 else "#fd7e14" if z > 1.96 else DARK_BLUE for z in z_vals]
    fig.add_trace(go.Bar(x=[str(d) for d in digits], y=z_vals, name="Z-Score",
        marker_color=z_colors, hovertemplate="Digit %{x}<br>Z = %{y:.3f}<extra></extra>",
        showlegend=False), row=1, col=2)
    for thr, col, lbl in [(1.96, "#fd7e14", "5% sig"), (2.576, RED, "1% sig")]:
        fig.add_hline(y=thr, line_dash="dash", line_color=col,
            annotation_text=lbl, row=1, col=2, annotation_font_color=col)
    fig.update_layout(**PL, title=dict(text=f"<b>{title}</b>", font=dict(color=GOLD, size=14)),
        barmode="group", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.25, font=dict(color=TXT)))
    fig.update_xaxes(title_text="Leading Digit", title_font_color=MUTED, gridcolor="#2a3f5f", linecolor="#2a3f5f")
    fig.update_yaxes(title_font_color=MUTED, gridcolor="#2a3f5f", linecolor="#2a3f5f")
    return fig

def chart_deviation(result):
    digits = list(range(1, 10))
    devs = [(result["obs_p"][d] - result["exp_p"][d])*100 for d in digits]
    colors = [RED if d > 1.5 else GREEN if d < -1.5 else MID_BLUE for d in devs]
    fig = go.Figure(go.Bar(x=[str(d) for d in digits], y=devs, marker_color=colors,
        hovertemplate="Digit %{x}<br>Deviation: %{y:+.2f}%<extra></extra>",
        text=[f"{v:+.1f}%" for v in devs], textposition="outside", textfont=dict(color=TXT, size=11)))
    fig.add_hline(y=0, line_color="white", line_width=1.5)
    fig.add_hline(y=1.5, line_dash="dash", line_color=RED, line_width=1, annotation_text="Alert +1.5%", annotation_font_color=RED)
    fig.add_hline(y=-1.5, line_dash="dash", line_color=GOLD, line_width=1, annotation_text="Alert -1.5%", annotation_font_color=GOLD)
    fig.update_layout(**PL, title=dict(text="<b>Deviation from Benford's Law (%)</b>", font=dict(color=GOLD, size=14)),
        xaxis_title="Leading Digit", yaxis_title="Observed − Expected (%)", height=360)
    return fig

def chart_last_digit(probs):
    digits = list(range(10))
    vals = [probs.get(d, 0)*100 for d in digits]
    colors = [RED if d in (0, 5) else DARK_BLUE for d in digits]
    fig = go.Figure(go.Bar(x=[str(d) for d in digits], y=vals, marker_color=colors,
        hovertemplate="Last digit %{x}<br>%{y:.1f}%<extra></extra>",
        text=[f"{v:.1f}%" for v in vals], textposition="outside", textfont=dict(color=TXT, size=10)))
    fig.add_hline(y=10, line_dash="dash", line_color=GOLD, annotation_text="Expected 10%", annotation_font_color=GOLD)
    fig.update_layout(**PL, title=dict(text="<b>Last-Digit Distribution (Round-Number Test)</b>",
        font=dict(color=GOLD, size=14)), xaxis_title="Last Digit", yaxis_title="Frequency (%)", height=360)
    return fig

def chart_log_histogram(data):
    data = np.array(data, dtype=float); data = data[data > 0]
    log_data = np.log10(data)
    fig = go.Figure(go.Histogram(x=log_data, nbinsx=60, marker_color=MID_BLUE,
        marker_line=dict(color=DARK_BLUE, width=0.3),
        hovertemplate="log₁₀(Amount): %{x:.2f}<br>Count: %{y}<extra></extra>"))
    for d in range(1, 10):
        fig.add_vline(x=np.log10(d), line_color=GOLD, line_width=0.6, opacity=0.5)
    fig.update_layout(**PL, title=dict(text="<b>Transaction Amount Distribution (Log₁₀ Scale)</b>",
        font=dict(color=GOLD, size=14)), xaxis_title="log₁₀(Amount) — Gold lines mark digit boundaries",
        yaxis_title="Count", height=350)
    return fig

# ─────────────────────────────────────────────────────────────
# SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────
def show_stats_panel(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample Size",  f"{result['n']:,}")
    c2.metric("Chi-Squared",  f"{result['chi2']:.2f}", delta=f"p = {result['p_chi2']:.4f}", delta_color="inverse")
    c3.metric("MAD",          f"{result['mad']:.4f}", delta="Threshold: 0.015", delta_color="off")
    c4.metric("KS Statistic", f"{result['ks']:.4f}")
    css = {"ok":"verdict-ok","warn":"verdict-warn","bad":"verdict-bad"}
    st.markdown(f'<div class="{css[result["level"]]}">🔍 {result["verdict"]}</div>', unsafe_allow_html=True)

def show_digit_table(result):
    rows = []
    for d in range(1, 10):
        z = result["z"][d]
        flag = "🚨 ***" if z > 2.576 else "⚠️ **" if z > 1.96 else "⚡ *" if z > 1.645 else "✅"
        rows.append({"Digit": d, "Observed": result["counts"][d],
            "Expected": f"{result['expected'][d]:.1f}", "Obs %": f"{result['obs_p'][d]*100:.2f}%",
            "Benford %": f"{result['exp_p'][d]*100:.2f}%",
            "Deviation": f"{(result['obs_p'][d]-result['exp_p'][d])*100:+.2f}%",
            "Z-Score": f"{z:.3f}", "Flag": flag})
    st.dataframe(pd.DataFrame(rows).set_index("Digit"), use_container_width=True)

def dl_btn(csv_bytes, filename, label="⬇️  Download CSV"):
    st.download_button(label=label, data=csv_bytes, file_name=filename,
                       mime="text/csv", use_container_width=True)

def dl_info(rows, cols, note):
    st.markdown(f"""
    <div class="mp-card-blue" style="padding:12px 16px;margin-bottom:12px;">
      <div style="font-size:0.78rem;color:{MUTED};margin-bottom:4px;
           text-transform:uppercase;letter-spacing:0.05em;">Dataset Info</div>
      <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:6px;">
        <span style="font-size:0.85rem;color:{TXT};">📏 <b style="color:{GOLD};">{rows}</b> rows</span>
        <span style="font-size:0.85rem;color:{TXT};">🗂️ <b style="color:{GOLD};">{cols}</b> columns</span>
      </div>
      <div style="font-size:0.82rem;color:{LIGHT_BLUE};">ℹ️ {note}</div>
    </div>""", unsafe_allow_html=True)

def col_dict_table(col_defs):
    """Render column definition table from list of (col, type, desc) tuples."""
    st.dataframe(pd.DataFrame(
        [{"Column": c, "Type": t, "Description": d} for c, t, d in col_defs]
    ), use_container_width=True, hide_index=True)

def zip_three(df1, name1, df2, name2, df3, name3, readme=""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name1, df1.to_csv(index=False))
        zf.writestr(name2, df2.to_csv(index=False))
        zf.writestr(name3, df3.to_csv(index=False))
        if readme:
            zf.writestr("README.txt", readme)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────
# PRE-BUILD ALL THREE CASE DATASETS (deterministic, same seeds)
# ─────────────────────────────────────────────────────────────

# ── Case 1: GST Invoice Fraud ─────────────────────────────────
np.random.seed(101)
_fraud_invoices = np.concatenate([
    np.random.uniform(50000, 59999, 80),
    np.random.uniform(90000, 99000, 60),
    np.random.choice([25000, 50000, 100000, 200000], 40),
    np.random.uniform(10000, 19999, 20),
])
_supplier_pool = np.random.choice(["Shree Fab", "Aarav Traders", "Lotus Tex"], len(_fraud_invoices))
_invoice_dates = pd.date_range("2023-04-01", periods=len(_fraud_invoices), freq="B")
_fd1 = [first_digit(x) for x in _fraud_invoices]
_ld1 = [int(str(int(abs(x)))[-1]) for x in _fraud_invoices]

DF_CASE1 = pd.DataFrame({
    "Invoice_Number":     [f"INV-{i+1:03d}" for i in range(len(_fraud_invoices))],
    "Invoice_Date":       _invoice_dates.strftime("%Y-%m-%d"),
    "Supplier":           _supplier_pool,
    "Invoice_Amount_INR": np.round(_fraud_invoices, 2),
    "GST_Rate_Pct":       np.random.choice([5, 12, 18], len(_fraud_invoices)),
    "ITC_Claimed_INR":    np.round(_fraud_invoices * np.random.choice([0.05, 0.12, 0.18], len(_fraud_invoices)), 2),
    "First_Digit":        _fd1,
    "Last_Digit":         _ld1,
    "Benford_Expected_Pct": [round(BENFORD.get(d, 0)*100, 2) for d in _fd1],
    "Fraud_Flag":         1,
})

# ── Case 2: Expense Fraud ────────────────────────────────────
np.random.seed(202)
_team_clean = np.concatenate([10**np.random.uniform(2, 3.7, 900), np.random.exponential(2000, 300)])
_sm_verma   = np.concatenate([np.random.uniform(4200, 4999, 120),
                               np.random.choice([4500, 4800, 4999, 4950], 80),
                               np.random.uniform(100, 3000, 50)])
_employees_clean = np.random.choice(["R Patel","A Kumar","P Singh","N Mehta","K Sharma",
                                      "S Gupta","M Iyer","D Rao","B Joshi"], len(_team_clean))
_emp_labels = np.concatenate([_employees_clean, np.full(len(_sm_verma), "SM Verma")])
_amounts_all = np.concatenate([_team_clean, _sm_verma])
_claim_dates = pd.date_range("2023-04-01", periods=len(_amounts_all), freq="B")
_fd2 = [first_digit(x) for x in _amounts_all]
_ld2 = [int(str(int(abs(x)))[-1]) for x in np.maximum(_amounts_all, 1)]

DF_CASE2 = pd.DataFrame({
    "Claim_ID":           [f"CLM-{i+1:04d}" for i in range(len(_amounts_all))],
    "Claim_Date":         _claim_dates.strftime("%Y-%m-%d"),
    "Employee":           _emp_labels,
    "Claim_Amount_INR":   np.round(_amounts_all, 2),
    "Expense_Category":   np.random.choice(["Travel","Meals","Accommodation","Fuel","Miscellaneous"], len(_amounts_all)),
    "First_Digit":        _fd2,
    "Last_Digit":         _ld2,
    "Benford_Expected_Pct": [round(BENFORD.get(d, 0)*100, 2) for d in _fd2],
    "Near_5k_Threshold":  [1 if 4000 <= x < 5000 else 0 for x in _amounts_all],
    "Fraud_Flag":         [1 if e == "SM Verma" else 0 for e in _emp_labels],
})

# ── Case 3: Bank Structuring ─────────────────────────────────
np.random.seed(404)
_deposits = np.concatenate([
    np.random.uniform(920000, 998000, 30),
    np.random.uniform(850000, 899000, 10),
    np.random.uniform(970000, 999000, 5),
])
_dep_dates  = pd.date_range("2024-01-02", periods=len(_deposits), freq="D")
_fd3 = [first_digit(x) for x in _deposits]
_ld3 = [int(str(int(abs(x)))[-1]) for x in _deposits]

DF_CASE3 = pd.DataFrame({
    "Transaction_ID":       [f"TXN-{i+1:03d}" for i in range(len(_deposits))],
    "Date":                 _dep_dates.strftime("%Y-%m-%d"),
    "Transaction_Type":     "Cash Deposit",
    "Amount_INR":           np.round(_deposits, 2),
    "Amount_Lakhs":         np.round(_deposits / 100000, 4),
    "Below_10L_Threshold":  [1 if x < 1000000 else 0 for x in _deposits],
    "First_Digit":          _fd3,
    "Last_Digit":           _ld3,
    "Benford_Expected_Pct": [round(BENFORD.get(d, 0)*100, 2) for d in _fd3],
    "CTR_Triggered":        [0 for _ in _deposits],
    "Structuring_Flag":     1,
    "Cumulative_Amount_INR": np.round(np.cumsum(_deposits), 2),
})

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:18px 10px 12px; background:#001428;
         border-radius:10px; margin-bottom:12px; border:1px solid {GOLD}55;">
      <div style="font-size:22px; color:{GOLD}; font-weight:900; letter-spacing:1px;">
        🏔️ THE MOUNTAIN PATH</div>
      <div style="font-size:11px; color:#ADD8E6; margin-top:4px; letter-spacing:2px; text-transform:uppercase;">
        World of Finance</div>
      <div style="height:2px; background:linear-gradient(90deg,transparent,{GOLD},transparent); margin:10px 0;"></div>
      <div style="font-size:12px; color:#ffffff; font-weight:600;">Prof. V. Ravichandran</div>
      <div style="font-size:11px; color:#ADD8E6; margin-top:3px;">
        <a href="https://themountainpathacademy.com" target="_blank"
           style="color:{GOLD} !important; text-decoration:none;">themountainpathacademy.com</a>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="font-size:11px; color:{GOLD}; font-weight:700;
         text-transform:uppercase; letter-spacing:1px; padding:4px 6px 6px; margin-bottom:4px;">
      📚 NAVIGATE</div>""", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠 Home", "📖 Learn: Benford's Law", "📊 Interactive Analyzer",
        "🏦 Case 1: GST Invoice Fraud", "💼 Case 2: Expense Report Fraud",
        "🏛️ Case 3: Bank Structuring", "🤖 ML Anomaly Detection", "❓ Quiz & Assessment"],
        label_visibility="collapsed")

    st.markdown(f"""
    <div style="margin-top:14px; padding:12px 14px; background:#001428;
         border-radius:8px; border:1px solid {GOLD}33;">
      <div style="font-size:11px; font-weight:700; color:{GOLD};
           text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">🔑 Key Topics</div>
      <div style="font-size:12px; color:#e6f1ff; line-height:1.9;">
        <span style="color:{GOLD};">▸</span> Benford's Law Formula<br>
        <span style="color:{GOLD};">▸</span> First &amp; Second Digit Tests<br>
        <span style="color:{GOLD};">▸</span> Chi-Squared &amp; MAD Tests<br>
        <span style="color:{GOLD};">▸</span> GST / Tax Fraud Detection<br>
        <span style="color:{GOLD};">▸</span> Expense Reimbursement Fraud<br>
        <span style="color:{GOLD};">▸</span> PMLA Structuring Detection<br>
        <span style="color:{GOLD};">▸</span> Isolation Forest ML<br>
        <span style="color:{GOLD};">▸</span> Ethical Considerations
      </div>
    </div>
    <div style="height:1px; background:{GOLD}33; margin:14px 0;"></div>
    <div style="font-size:11px; color:#ADD8E6; text-align:center; line-height:1.8;">
      <span style="color:#ffffff; font-weight:600;">© 2025 The Mountain Path</span><br>
      <a href="https://www.linkedin.com/in/trichyravis" target="_blank"
         style="color:{GOLD} !important; font-weight:600;">LinkedIn</a>
      <span style="color:{GOLD};">  |  </span>
      <a href="https://github.com/trichyravis" target="_blank"
         style="color:{GOLD} !important; font-weight:600;">GitHub</a>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(f"""
    <div class="hero-wrap">
      <div style="font-size:42px; color:{GOLD}; font-weight:900; letter-spacing:2px;">BENFORD'S LAW</div>
      <div style="font-size:22px; color:{TXT}; margin:6px 0;">Fraud Analytics &amp; Anomaly Detection</div>
      <div style="font-size:14px; color:{MUTED}; margin-top:10px;">
        An Interactive Learning Platform &nbsp;|&nbsp; The Mountain Path – World of Finance</div>
      <div style="margin-top:14px;">
        <span class="badge">Financial Analytics</span>
        <span class="badge">Fraud Detection</span>
        <span class="badge">Statistical Testing</span>
        <span class="badge">Machine Learning</span>
        <span class="badge">Python</span>
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Benford Formula", "P(d) = log₁₀(1+1/d)")
    c2.metric("Numbers Starting with 1", "30.1%", delta="Most common first digit", delta_color="off")
    c3.metric("Live Caselets", "3", delta="GST · Expense · Banking", delta_color="off")
    c4.metric("ML Method", "Isolation Forest", delta="Anomaly Detection", delta_color="off")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 📌 What You Will Learn")
        for icon, title, desc in [
            ("📐","The Mathematics","Understand Benford's Law from first principles"),
            ("📊","Statistical Tests","Chi-squared, MAD, Z-score and KS tests"),
            ("🏦","GST Fraud","Detect fictitious invoice fraud in GST returns"),
            ("💼","Expense Fraud","Identify inflated corporate expense claims"),
            ("🏛️","Bank Structuring","Catch PMLA threshold gaming in banking"),
            ("🤖","ML Methods","Isolation Forest for multivariate anomaly detection"),
            ("⚖️","Ethics","Responsible use of fraud analytics in practice"),
        ]:
            st.markdown(f'''<div style="background:#0d1b2e; border:1px solid #FFD70044;
                border-left:4px solid #FFD700; border-radius:10px; padding:14px 18px; margin-bottom:10px;">
              <span style="color:#FFD700; font-weight:700; font-size:14px;">{icon} {title}</span><br>
              <span style="color:#e6f1ff; font-size:13px;">{desc}</span>
            </div>''', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Benford's Law at a Glance")
        digits = list(range(1, 10)); probs = [BENFORD[d]*100 for d in digits]
        fig = go.Figure(go.Bar(x=[str(d) for d in digits], y=probs,
            marker_color=[DARK_BLUE if i < 3 else MID_BLUE for i in range(9)],
            marker_line=dict(color=GOLD, width=1),
            text=[f"{p:.1f}%" for p in probs], textposition="outside", textfont=dict(color=TXT, size=10)))
        fig.update_layout(**PL, title=dict(text="<b>Expected First-Digit Frequency</b>",
            font=dict(color=GOLD, size=13)), xaxis_title="Leading Digit",
            yaxis_title="Probability (%)", height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div class="formula-box">
          <div style="color:{MUTED}; font-size:11px; margin-bottom:6px;">BENFORD'S LAW FORMULA</div>
          <div style="color:{GOLD}; font-size:24px; font-weight:900; font-family:Georgia,serif;">
            P(d) = log<sub>10</sub>(1 + 1/d)</div>
          <div style="color:{LIGHT_BLUE}; font-size:11px; margin-top:6px;">d ∈ {{1,2,3,4,5,6,7,8,9}}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("""**How to Use:** Navigate using the sidebar. Start with *Learn: Benford's Law*,
    then explore the three live case studies (each with downloadable raw data), the interactive
    analyzer, and finally the ML module.""")


# ═════════════════════════════════════════════════════════════
# PAGE: LEARN
# ═════════════════════════════════════════════════════════════
elif page == "📖 Learn: Benford's Law":
    st.markdown("## 📖 Understanding Benford's Law")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📜 History & Concept","📐 Mathematics","📊 Extended Analysis",
        "🧪 Statistical Tests","⚠️ Limitations & Ethics"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🕰️ A Discovery from a Worn Logarithm Book")
            for title, body, cls in [
                ("1881 — Simon Newcomb",
                 "An American astronomer noticed early logarithm table pages were more worn — people looked up numbers starting with 1 far more often.",
                 "mp-card"),
                ("1938 — Frank Benford",
                 "Benford tested <b>20,229 data points</b> from 20 datasets: river areas, atomic weights, baseball stats, street addresses. Same pattern appeared every time.",
                 "mp-card"),
                ("The Core Insight",
                 "The leading digit is <em>not</em> uniformly distributed. Digit 1 appears ~30% — six times more often than digit 9 (≈4.6%).",
                 "mp-card-green"),
            ]:
                color = GOLD if cls == "mp-card" else GREEN
                st.markdown(f'<div class="{cls}"><b style="color:{color};">{title}</b><br>{body}</div>', unsafe_allow_html=True)

            st.markdown("### 📋 When Does Benford's Law Apply?")
            ca, cb = st.columns(2)
            with ca:
                st.success("✅ **Follows Benford's Law**")
                for i in ["Revenue and sales figures","Expense and invoice amounts","Tax return values",
                          "Financial statement line items","Stock prices and volumes","GST and customs values"]:
                    st.markdown(f"• {i}")
            with cb:
                st.error("❌ **Does NOT Follow**")
                for i in ["Phone numbers (fixed format)","Employee IDs (sequential)","Ages (bounded 0-120)",
                          "Psychologically anchored prices","Uniformly assigned numbers","Zip/Postal codes"]:
                    st.markdown(f"• {i}")

        with col2:
            st.markdown("### 📊 The Benford Distribution")
            df_ben = pd.DataFrame({"Digit": range(1, 10),
                "Probability (%)": [round(BENFORD[d]*100, 2) for d in range(1,10)],
                "1-in-N": [f"1 in {round(1/BENFORD[d])}" for d in range(1,10)]})
            st.dataframe(df_ben.set_index("Digit"), use_container_width=True)
            st.markdown(f"""<div class="formula-box" style="margin-top:16px;">
              <div style="color:{MUTED}; font-size:11px;">KEY FACT</div>
              <div style="color:{GOLD}; font-size:18px; font-weight:800; margin:8px 0;">
                Digits 1–3 account for<br><span style="font-size:36px;">60.2%</span><br>of all leading digits!</div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="formula-box">
              <div style="color:{MUTED}; font-size:12px; margin-bottom:8px;">BENFORD'S LAW — FIRST DIGIT</div>
              <div style="color:{GOLD}; font-size:28px; font-weight:900; font-family:Georgia,serif;">
                P(d) = log<sub>10</sub>(1 + 1/d)</div>
              <div style="color:{TXT}; font-size:13px; margin-top:10px;">where d ∈ {{1,2,3,4,5,6,7,8,9}}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="mp-card-blue">
              <b style="color:{LIGHT_BLUE};">Compound growth intuition:</b><br><br>
              Start with ₹100 (digit: <b style="color:{GOLD};">1</b>)<br>
              +10% each step ... stays at digit 1 for 8 steps!<br>
              At ₹214 → first digit becomes <b style="color:{GOLD};">2</b><br><br>
              <b>Moving 1→2 requires 100% increase. Moving 9→10 needs only 11%.</b>
            </div>""", unsafe_allow_html=True)

        with col2:
            sel = st.slider("Select a leading digit:", 1, 9, 1)
            prob = BENFORD[sel]
            st.markdown(f"""<div style="background:{CARD_BG}; border:2px solid {GOLD};
                border-radius:10px; padding:20px; text-align:center; margin:10px 0;">
              <div style="color:{MUTED}; font-size:12px;">Formula</div>
              <div style="color:{GOLD}; font-size:18px; font-family:Georgia,serif;">
                P({sel}) = log₁₀(1+1/{sel}) = log₁₀({sel+1}/{sel})</div>
              <div style="color:{TXT}; font-size:36px; font-weight:900; margin:12px 0;">{prob*100:.2f}%</div>
              <div style="color:{LIGHT_BLUE}; font-size:13px;">Appears 1 in {round(1/prob)} numbers</div>
            </div>""", unsafe_allow_html=True)
            cum, running = [], 0
            for d in range(1, 10):
                running += BENFORD[d]*100
                cum.append({"Digit": d, "Cumulative %": round(running, 2)})
            st.dataframe(pd.DataFrame(cum).set_index("Digit"), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Second-Digit Distribution")
            sdp = {d: sum(np.log10(1 + 1/(10*k+d)) for k in range(1,10)) for d in range(10)}
            st.dataframe(pd.DataFrame({"Digit": list(range(10)),
                "Expected %": [round(sdp[d]*100,2) for d in range(10)]}).set_index("Digit"), use_container_width=True)
        with c2:
            st.markdown(f"""<div class="mp-card-red">
              <b style="color:{RED};">Fraudsters Love Round Numbers!</b><br><br>
              In natural data, each last digit (0–9) appears ~10%.
              In fabricated data, digits 0 and 5 often appear 30–40%.<br><br>
              <b style="color:{GOLD};">Round Number Score = Observed % of 0s and 5s ÷ 20%</b><br>
              Score > 2.0 (>40%) is a significant red flag.
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="mp-card-blue">
          <b style="color:{LIGHT_BLUE};">The Summation Test:</b> Each two-digit group (10–99) should account
          for ~<b style="color:{GOLD};">1/90 ≈ 1.11%</b> of total value. Any group with ratio >2× is flagged.
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown(f"""<div class="mp-card">
          <b style="color:{GOLD};">Why statistical tests?</b> Observed frequencies never perfectly match
          Benford's Law. Tests tell us: is this deviation suspicious or just random chance?
        </div>""", unsafe_allow_html=True)
        t1,t2,t3,t4 = st.tabs(["Chi-Squared","MAD Test","Z-Score","KS Test"])
        with t1:
            st.markdown(f"""<div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">CHI-SQUARED GOODNESS-OF-FIT</div>
              <div style="color:{GOLD}; font-size:22px; font-family:Georgia,serif;">χ² = Σ (Oᵈ − Eᵈ)² / Eᵈ</div>
              <div style="color:{TXT}; font-size:12px; margin-top:8px;">Degrees of freedom = 8</div>
            </div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Significance Level":["10%","5%","1%","0.1%"],
                "Critical Value (df=8)":[13.36,15.51,20.09,26.12],
                "Action":["Review","Investigate","Urgent","Immediate Alert"]}),
                use_container_width=True, hide_index=True)
            st.warning("⚠️ With n > 5,000, tiny deviations become significant. Always combine with MAD.")
        with t2:
            st.markdown(f"""<div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">MEAN ABSOLUTE DEVIATION</div>
              <div style="color:{GOLD}; font-size:22px; font-family:Georgia,serif;">MAD = (1/9) × Σ |Pᵈobs − Pᵈbenford|</div>
            </div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"MAD Value":["0.000–0.006","0.006–0.012","0.012–0.015",">0.015"],
                "Interpretation":["Close conformity","Acceptable","Marginal","Non-conformity"],
                "Action":["No issue ✅","Monitor 👀","Review ⚠️","INVESTIGATE 🚨"]}),
                use_container_width=True, hide_index=True)
            st.success("✅ MAD is not inflated by large sample sizes — the preferred metric (Nigrini, 2012).")
        with t3:
            st.markdown(f"""<div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">Z-SCORE PER DIGIT</div>
              <div style="color:{GOLD}; font-size:18px; font-family:Georgia,serif;">
                Zᵈ = (|Pᵈobs − Pᵈbenford| − 1/(2n)) / √(Pᵈbenford(1−Pᵈbenford)/n)</div>
              <div style="color:{TXT}; font-size:11px; margin-top:6px;">Flag: |Z| > 1.96 (5%) or |Z| > 2.576 (1%)</div>
            </div>""", unsafe_allow_html=True)
            st.info("💡 Identifies WHICH specific digit is anomalous.")
        with t4:
            st.markdown(f"""<div class="formula-box">
              <div style="color:{MUTED}; font-size:11px;">KOLMOGOROV-SMIRNOV TEST</div>
              <div style="color:{GOLD}; font-size:22px; font-family:Georgia,serif;">D = max |Fobs(d) − FBenford(d)|</div>
            </div>""", unsafe_allow_html=True)
            st.info("💡 Best for small samples (< 100 records).")

    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚠️ Critical Limitations")
            for title, desc in [
                ("Not all data qualifies","Phone numbers, ages, anchored prices do not follow Benford."),
                ("Sample size sensitivity","Chi-squared oversensitive at n>5,000. Prefer MAD."),
                ("Benford ≠ innocence","Sophisticated fraudsters can fabricate conforming numbers."),
                ("Industry deviations","FMCG prices at ₹X9.99; real estate near stamp duty thresholds."),
                ("Minimum sample","At least 50–100 records; ideally 1,000+."),
            ]:
                st.markdown(f'<div class="mp-card-red"><b style="color:{RED};">{title}</b><br>{desc}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("### ⚖️ Ethical Principles")
            for title, desc in [
                ("Presumption of innocence","A flag is a signal for investigation, NOT evidence of fraud."),
                ("Algorithmic fairness","Audit ML models for demographic bias."),
                ("Data privacy","Comply with DPDP Act 2023 and be proportionate to risk."),
                ("Transparency","Individuals have the right to understand adverse decisions."),
                ("Human-in-the-loop","Machines flag; humans decide. High-stakes actions need human review."),
                ("Model governance","Re-validate periodically. Fraudsters adapt."),
            ]:
                st.markdown(f'<div class="mp-card-green"><b style="color:{GREEN};">{title}</b><br>{desc}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: INTERACTIVE ANALYZER
# ═════════════════════════════════════════════════════════════
elif page == "📊 Interactive Analyzer":
    st.markdown("## 📊 Interactive Benford Analyzer")
    st.markdown("Upload your own data or use the synthetic generator to explore Benford's Law in action.")

    tab_upload, tab_gen = st.tabs(["📁 Upload / Paste Data","🎲 Synthetic Data Generator"])

    with tab_upload:
        col1, col2 = st.columns([1, 2])
        with col1:
            method = st.radio("Input method:", ["Paste numbers","Upload CSV"])
            if method == "Paste numbers":
                raw = st.text_area("Paste numbers (one per line or comma-separated):",
                    "145230\n23400\n8750\n312000\n67800\n190000\n445000\n28900\n"
                    "73500\n156000\n92000\n34500\n228000\n87600\n143000", height=200)
                if st.button("Analyse", type="primary"):
                    nums = []
                    for token in raw.replace(",","\n").split():
                        try: nums.append(float(token))
                        except: pass
                    st.session_state["analyzer_data"] = nums
            else:
                uploaded = st.file_uploader("Upload CSV file:", type=["csv"])
                if uploaded:
                    df_up = pd.read_csv(uploaded)
                    nc = df_up.select_dtypes(include=np.number).columns.tolist()
                    cs = st.selectbox("Select numeric column:", nc)
                    if st.button("Analyse Column", type="primary"):
                        st.session_state["analyzer_data"] = df_up[cs].dropna().tolist()

        with col2:
            if "analyzer_data" in st.session_state:
                data = st.session_state["analyzer_data"]
                result = benford_analysis(data)
                if result:
                    show_stats_panel(result)
                    st.plotly_chart(chart_first_digit(result, "Your Data — Benford Analysis"), use_container_width=True)
                    show_digit_table(result)
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(chart_deviation(result), use_container_width=True)
                    with c2:
                        lc, lp, rp = last_digit_analysis(data)
                        st.plotly_chart(chart_last_digit(lp), use_container_width=True)
                        if rp > 30: st.error(f"🚨 Round Number Score: {rp:.1f}% of last digits are 0 or 5 (expected: 20%). Suspicious!")
                        else: st.success(f"✅ Round Number Score: {rp:.1f}% (normal range)")
                    st.plotly_chart(chart_log_histogram(data), use_container_width=True)

    with tab_gen:
        col1, col2 = st.columns([1, 2])
        with col1:
            n_clean = st.slider("Clean transactions:", 500, 5000, 2000, 100)
            n_fraud = st.slider("Fraudulent transactions:", 0, 500, 150, 10)
            ft = st.selectbox("Fraud pattern:", ["Structuring (just-below threshold)",
                "Round numbers (₹10k, ₹25k, ₹50k)","Digit-5 inflation","No fraud (clean dataset)"])
            if st.button("Generate & Analyse", type="primary"):
                np.random.seed(42)
                clean = np.concatenate([10**np.random.uniform(2,6,int(n_clean*0.7)),
                                        np.random.exponential(25000, int(n_clean*0.3))])
                if ft == "No fraud (clean dataset)": synth = clean
                elif "Structuring" in ft: synth = np.concatenate([clean, np.random.uniform(950000,999000,n_fraud)])
                elif "Round" in ft: synth = np.concatenate([clean, np.random.choice([10000,25000,50000,100000],n_fraud)])
                else: synth = np.concatenate([clean, np.random.uniform(50000,59999,n_fraud)])
                st.session_state["synth_data"] = synth; st.session_state["synth_type"] = ft

        with col2:
            if "synth_data" in st.session_state:
                result = benford_analysis(st.session_state["synth_data"])
                if result:
                    st.caption(f"📊 Analysis: {st.session_state.get('synth_type','')}")
                    show_stats_panel(result)
                    st.plotly_chart(chart_first_digit(result, f"Synthetic: {st.session_state.get('synth_type','')}"), use_container_width=True)
                    show_digit_table(result)


# ═════════════════════════════════════════════════════════════
# CASE 1: GST INVOICE FRAUD
# ═════════════════════════════════════════════════════════════
elif page == "🏦 Case 1: GST Invoice Fraud":
    st.markdown("## 🏦 Case Study 1: GST Invoice Fraud Detection")
    st.markdown(f"""<div class="hero-wrap" style="padding:20px 28px; text-align:left;">
      <span class="badge">Case Study 1</span>
      <span class="badge-red badge">FRAUD DETECTED</span>
      <h3 style="color:{GOLD}; margin:10px 0 6px;">Fictitious Invoice Fraud in GST Returns</h3>
      <p style="color:{TXT}; margin:0;"><b>Scenario:</b> A textile trader in Mumbai files GST returns claiming ₹5 crore in
        Input Tax Credit (ITC) based on 200 purchase invoices from three suppliers.
        The GSTN analytics team runs a Benford's Law test as part of routine screening.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Claimed ITC", "₹5,00,00,000", delta="Fraudulent")
    col2.metric("Invoices Scrutinised", "200")
    col3.metric("Suppliers Involved", "3 (shell entities)")

    # Use pre-built dataset
    data = DF_CASE1["Invoice_Amount_INR"].values
    st.session_state["case1_data"] = data
    result = benford_analysis(data)

    tab1, tab2, tab3, tab4, tab_dl = st.tabs([
        "📋 Background", "📊 Benford Analysis",
        "🔍 Investigation Findings", "📚 Learning Points", "📥 Download Raw Data"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""<div class="mp-card">
              <b style="color:{GOLD};">Business Context</b><br><br>
              M/s Radha Textile Traders filed GSTR-3B for FY2023-24 claiming substantial ITC.
              Risk-scoring flagged: ITC-to-turnover ratio 340% above industry average,
              all three suppliers registered within 6 months, invoice amounts unusually concentrated.
            </div>
            <div class="mp-card-red">
              <b style="color:{RED};">Known Fraud Behaviour</b><br><br>
              Fabricators tend to: use round amounts (₹25k, ₹50k, ₹1L),
              cluster near reporting thresholds, underuse digit 1, overuse digits 5–9.
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("**Sample Invoice Amounts (₹):**")
            st.dataframe(DF_CASE1[["Invoice_Number","Invoice_Date","Supplier","Invoice_Amount_INR"]].head(15),
                         use_container_width=True, hide_index=True)

    with tab2:
        if result:
            show_stats_panel(result)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(chart_first_digit(result, "GST Invoice Benford Analysis"), use_container_width=True)
            with col2: st.plotly_chart(chart_deviation(result), use_container_width=True)
            show_digit_table(result)
            lc, lp, rp = last_digit_analysis(data)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(chart_last_digit(lp), use_container_width=True)
            with col2:
                st.markdown(f"""<div class="mp-card-red" style="margin-top:20px;">
                  <b style="color:{RED}; font-size:16px;">🚨 Round Number Alert</b><br><br>
                  <b style="color:{GOLD}; font-size:24px;">{rp:.1f}%</b> of invoice amounts end in 0 or 5<br>
                  <span style="color:{MUTED};">(Expected: ~20% in natural data)</span><br><br>
                  Round Number Score: <b style="color:{RED};">{rp/20:.1f}×</b> the expected value
                </div>""", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if result:
                st.markdown(f"""<div class="mp-card-red">
                  <b style="color:{RED};">Benford Signals That Triggered Investigation</b>
                  <ul style="margin:10px 0; padding-left:20px; color:{TXT};">
                    <li>Digit-1: 10% (expected 30.1%) — massive under-representation</li>
                    <li>Digit-5: ~40% (expected 7.9%) — massive over-representation</li>
                    <li>Digit-9: ~30% (expected 4.6%) — extreme structuring signal</li>
                    <li>MAD = {result['mad']:.4f} — far above 0.015 threshold</li>
                    <li>χ² = {result['chi2']:.1f} — overwhelmingly rejects Benford</li>
                  </ul>
                </div>
                <div class="mp-card-red">
                  <b style="color:{RED};">What Officers Found on Physical Verification</b>
                  <ul style="margin:10px 0; padding-left:20px; color:{TXT};">
                    <li>All three "suppliers" — same registered address (vacant plot)</li>
                    <li>No physical inventory; no e-way bills matched</li>
                    <li>Bank accounts linked to same beneficial owner</li>
                    <li>₹5 Cr ITC fraud confirmed; criminal prosecution initiated</li>
                  </ul>
                </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="mp-card-green">
              <b style="color:{GREEN};">What Legitimate Textile Invoices Look Like</b><br><br>
              Real invoices: Qty × Rate × (1+GST%) — creating non-round amounts like ₹145.50/metre.
              Natural digit variation. MAD &lt; 0.010.
            </div>""", unsafe_allow_html=True)
            if result:
                st.dataframe(pd.DataFrame({
                    "Metric":["Detection time","ITC demand raised","Penalty (200%)","Criminal referral","Suppliers blacklisted"],
                    "Outcome":["< 2 hours (analytics)","₹5,00,00,000","₹10,00,00,000","Yes — CGST Act Sec 132","3 entities"]}),
                    use_container_width=True, hide_index=True)

    with tab4:
        if result:
            for i,(title,content) in enumerate([
                ("Why GST Fraud Fails Benford","Fabricated invoices cluster unnaturally — humans choose amounts mentally, not through real calculations."),
                ("The Digit-9 Structuring Signal","Invoices clustering at ₹90k–₹99k signal deliberate threshold gaming."),
                ("MAD as the Primary Test", f"MAD of {result['mad']:.4f} with 200 invoices is highly meaningful."),
                ("Combine with Other Signals","Benford flags; e-way bills + address + bank linkage confirm."),
                ("Scale of Application","GSTN screens millions of returns in minutes using this test."),
            ], 1):
                st.markdown(f"""<div class="mp-card-blue">
                  <b style="color:{LIGHT_BLUE};">Learning {i}: {title}</b><br>{content}
                </div>""", unsafe_allow_html=True)

    # ── DOWNLOAD TAB ────────────────────────────────────────
    with tab_dl:
        st.markdown(f"""<div class="mp-card">
        <b style="color:{GOLD};">📥 Case 1 Raw Dataset — GST Invoice Fraud (M/s Radha Textile Traders)</b><br><br>
        The complete 200-invoice dataset used in this analysis. Includes invoice metadata,
        amounts, GST fields, first/last digits, Benford expected probabilities, and fraud flags.
        Use this dataset to practise Benford analysis, round-number detection, and threshold gaming analysis.
        </div>""", unsafe_allow_html=True)

        dl_info(len(DF_CASE1), len(DF_CASE1.columns),
                "Simulated GST invoice data · Seed 101 · Fabric/textile trade · FY2023-24")

        st.dataframe(DF_CASE1, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            dl_btn(DF_CASE1.to_csv(index=False).encode("utf-8"),
                   "case1_gst_invoice_fraud_data.csv",
                   "⬇️  Download Case 1 CSV — GST Invoices (200 rows)")
        with col2:
            st.markdown(f"""<div class="mp-card-blue" style="padding:12px 16px;">
            <b style="color:{LIGHT_BLUE};">Suggested Exercises:</b><br>
            <span style="font-size:0.85rem;">
            1. Run benford_analysis() on Invoice_Amount_INR — verify MAD and Chi-squared<br>
            2. Plot first-digit distribution vs Benford expected<br>
            3. Filter by Supplier and compare digit patterns per supplier<br>
            4. Count round-number amounts (Last_Digit in 0,5) — what % do you find?
            </span></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_dict_table([
            ("Invoice_Number",    "string",      "Sequential invoice identifier (INV-001 to INV-200)"),
            ("Invoice_Date",      "date",        "Invoice date — business days, Apr 2023 onwards"),
            ("Supplier",         "string",      "Supplier name — 3 shell entities"),
            ("Invoice_Amount_INR","float",       "Invoice face value in Indian Rupees (₹)"),
            ("GST_Rate_Pct",     "int",         "GST rate applied: 5%, 12%, or 18%"),
            ("ITC_Claimed_INR",  "float",       "Input Tax Credit claimed (Amount × GST rate)"),
            ("First_Digit",      "int (1-9)",   "Leading digit of invoice amount"),
            ("Last_Digit",       "int (0-9)",   "Last digit — high 0s/5s indicate fabrication"),
            ("Benford_Expected_Pct","float",    "Expected % for this first digit under Benford's Law"),
            ("Fraud_Flag",       "int (0/1)",   "1 = fraudulent invoice in this simulation"),
        ])


# ═════════════════════════════════════════════════════════════
# CASE 2: EXPENSE FRAUD
# ═════════════════════════════════════════════════════════════
elif page == "💼 Case 2: Expense Report Fraud":
    st.markdown("## 💼 Case Study 2: Corporate Expense Report Fraud")
    st.markdown(f"""<div class="hero-wrap" style="padding:20px 28px; text-align:left;">
      <span class="badge">Case Study 2</span>
      <span class="badge-red badge">FRAUD DETECTED</span>
      <h3 style="color:{GOLD}; margin:10px 0 6px;">Fabricated Expense Reimbursement Fraud</h3>
      <p style="color:{TXT}; margin:0;"><b>Scenario:</b> A pan-India FMCG company processes 6,000 expense claims annually.
        Internal audit deploys Benford's Law analytics. One regional sales manager's data raises major red flags.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims Analysed", "6,000")
    col2.metric("Approval Threshold", "₹5,000 (Manager)")
    col3.metric("Suspected Fraudster", "1 Regional Sales Manager")

    # Retrieve pre-built data
    clean_data = DF_CASE2[DF_CASE2["Employee"] != "SM Verma"]["Claim_Amount_INR"].values
    fraud_data  = DF_CASE2[DF_CASE2["Employee"] == "SM Verma"]["Claim_Amount_INR"].values
    st.session_state["case2_clean"] = clean_data
    st.session_state["case2_fraud"] = fraud_data
    r_clean = benford_analysis(clean_data)
    r_fraud = benford_analysis(fraud_data)

    tab1, tab2, tab3, tab4, tab_dl = st.tabs([
        "📋 Background", "📊 Benford Analysis",
        "🔍 Per-Employee Analysis", "📚 Learning Points", "📥 Download Raw Data"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""<div class="mp-card">
              <b style="color:{GOLD};">Expense Claim Policy</b><br><br>
              ≤ ₹5,000: Self-approved · ₹5,001–₹25,000: Manager · > ₹25,000: VP Finance + docs.
              A fraudulent employee would keep claims just below ₹5,000.
            </div>
            <div class="mp-card-red">
              <b style="color:{RED};">Threshold Gaming Hypothesis</b><br><br>
              Spike in digit 4 (₹4,000–₹4,999), under-represented digits 1–2,
              last digits 0 and 5 (round amounts), high claim volume vs peers.
            </div>""", unsafe_allow_html=True)
        with col2:
            emp_summary = DF_CASE2.groupby("Employee").agg(
                Claims=("Claim_ID","count"),
                Avg_Amount=("Claim_Amount_INR","mean")).reset_index()
            emp_summary["Avg_Amount"] = emp_summary["Avg_Amount"].round(0).astype(int)
            emp_summary = emp_summary.sort_values("Claims", ascending=False).head(10)
            st.dataframe(emp_summary, use_container_width=True, hide_index=True)
            st.error("🚨 SM Verma: 250 claims vs average of ~38 for peers — immediate red flag!")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Team Average (Clean Data)")
            if r_clean:
                show_stats_panel(r_clean)
                st.plotly_chart(chart_first_digit(r_clean, "Team Expenses — Clean"), use_container_width=True)
        with col2:
            st.markdown("#### 🚨 SM Verma's Claims (Fraudulent)")
            if r_fraud:
                show_stats_panel(r_fraud)
                st.plotly_chart(chart_first_digit(r_fraud, "SM Verma — Suspicious"), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if r_clean: st.plotly_chart(chart_deviation(r_clean), use_container_width=True)
        with col2:
            if r_fraud: st.plotly_chart(chart_deviation(r_fraud), use_container_width=True)

        st.markdown("### 🔍 Digit-4 Deep Dive: Threshold Gaming Evidence")
        col1, col2 = st.columns(2)
        with col1:
            thresh_amounts = fraud_data[(fraud_data >= 4000) & (fraud_data < 5000)]
            fig_th = go.Figure(go.Histogram(x=thresh_amounts, nbinsx=40, marker_color=RED, opacity=0.8))
            fig_th.add_vline(x=5000, line_color=GOLD, line_width=2.5,
                             annotation_text="₹5,000 limit", annotation_font_color=GOLD)
            fig_th.update_layout(**PL, title=dict(text="<b>Claims ₹4,000–₹5,000</b>",
                font=dict(color=RED, size=13)), height=300)
            st.plotly_chart(fig_th, use_container_width=True)
        with col2:
            pct = len(thresh_amounts)/len(fraud_data)*100 if len(fraud_data) > 0 else 0
            if r_fraud:
                st.markdown(f"""<div class="mp-card-red" style="margin-top:10px;">
                  <b style="color:{RED};">🚨 Threshold Gaming Confirmed</b><br><br>
                  <b style="font-size:22px; color:{GOLD};">{pct:.0f}%</b> of SM Verma's claims fall ₹4k–₹4,999<br>
                  (peer rate: ~8%) — <b>6× the peer rate</b><br><br>
                  MAD = {r_fraud['mad']:.4f} (non-conforming) → strong grounds for HR investigation.
                </div>""", unsafe_allow_html=True)

    with tab3:
        st.info("Per-employee Benford screening — the most powerful application for expense fraud detection.")
        np.random.seed(303)
        emp_rows = []
        for name, edata in {
            "SM Verma": np.concatenate([np.random.uniform(4200,4999,120), np.random.choice([4500,4800,4999],80), np.random.uniform(100,2000,50)]),
            "R Patel":  10**np.random.uniform(2,3.7,42), "A Kumar": 10**np.random.uniform(2,3.6,38),
            "P Singh":  10**np.random.uniform(2,3.8,45), "N Mehta": 10**np.random.uniform(2,3.5,33),
            "K Sharma": np.concatenate([np.random.uniform(24000,24999,20), 10**np.random.uniform(2,3.7,21)]),
            "S Gupta":  10**np.random.uniform(2,3.6,37), "M Iyer":  10**np.random.uniform(2,3.9,29),
        }.items():
            r = benford_analysis(edata)
            if r:
                emp_rows.append({"Employee":name, "Claims":r["n"], "MAD":f"{r['mad']:.4f}",
                    "Chi-Sq":f"{r['chi2']:.1f}", "Verdict":r["verdict"][:25]+"...",
                    "Risk":"🚨 HIGH" if r["mad"]>0.015 else "⚠️ MED" if r["mad"]>0.008 else "✅ LOW"})
        st.dataframe(pd.DataFrame(emp_rows), use_container_width=True, hide_index=True)

    with tab4:
        if r_fraud:
            for i,(title,content) in enumerate([
                ("Per-Employee Testing","Individual fraudster invisible in aggregate but clear in isolation."),
                ("Threshold Gaming = Digit-4 Spike","Digit-4 over-rep at ₹5k threshold confirms gaming."),
                ("Volume is a Pre-Screen","250 vs 38 average claims is itself anomalous."),
                ("Last-Digit Test","Round amounts (₹4,500; ₹4,800; ₹4,950) — last-digit spikes confirm fabrication."),
                ("Human Review Non-Negotiable",f"MAD={r_fraud['mad']:.4f} is damning, but action needs due process."),
            ], 1):
                st.markdown(f'<div class="mp-card-blue"><b style="color:{LIGHT_BLUE};">Learning {i}: {title}</b><br>{content}</div>', unsafe_allow_html=True)

    # ── DOWNLOAD TAB ────────────────────────────────────────
    with tab_dl:
        st.markdown(f"""<div class="mp-card">
        <b style="color:{GOLD};">📥 Case 2 Raw Dataset — Corporate Expense Report Fraud</b><br><br>
        Complete expense claim dataset for all employees (1,250 claims). Includes SM Verma's
        fraudulent threshold-gaming claims alongside clean peer data. Contains fraud flags,
        threshold proximity indicators, and first/last digit columns for Benford analysis.
        </div>""", unsafe_allow_html=True)

        dl_info(len(DF_CASE2), len(DF_CASE2.columns),
                "Simulated expense claim data · Seed 202 · FMCG sales team · FY2023-24")

        # Show summary first, then full data
        summary = DF_CASE2.groupby("Employee").agg(
            Claims=("Claim_ID","count"), Total_INR=("Claim_Amount_INR","sum"),
            Avg_INR=("Claim_Amount_INR","mean"), Near_5k=("Near_5k_Threshold","sum"),
            Fraud_Claims=("Fraud_Flag","sum")).reset_index()
        summary["Total_INR"] = summary["Total_INR"].round(0).astype(int)
        summary["Avg_INR"]   = summary["Avg_INR"].round(0).astype(int)
        st.markdown(f"##### Employee Summary ({len(summary)} employees)")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown(f"##### Full Dataset (first 30 rows shown)")
        st.dataframe(DF_CASE2.head(30), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            dl_btn(DF_CASE2.to_csv(index=False).encode("utf-8"),
                   "case2_expense_fraud_data.csv",
                   "⬇️  Download Case 2 CSV — Expense Claims (1,250 rows)")
        with col2:
            dl_btn(summary.to_csv(index=False).encode("utf-8"),
                   "case2_expense_fraud_summary.csv",
                   "⬇️  Download Case 2 Summary CSV — By Employee")

        st.markdown("<br>", unsafe_allow_html=True)
        col_dict_table([
            ("Claim_ID",            "string",    "Sequential claim identifier (CLM-0001 to CLM-1250)"),
            ("Claim_Date",          "date",      "Claim submission date — business days"),
            ("Employee",            "string",    "Employee name — SM Verma is the fraudster"),
            ("Claim_Amount_INR",    "float",     "Claimed expense amount in Indian Rupees (₹)"),
            ("Expense_Category",    "string",    "Travel / Meals / Accommodation / Fuel / Miscellaneous"),
            ("First_Digit",         "int (1-9)", "Leading digit of claim amount"),
            ("Last_Digit",          "int (0-9)", "Last digit — round amounts show 0/5 spikes"),
            ("Benford_Expected_Pct","float",     "Expected % for this first digit under Benford's Law"),
            ("Near_5k_Threshold",   "int (0/1)", "1 if claim falls in ₹4,000–₹4,999 (threshold gaming zone)"),
            ("Fraud_Flag",          "int (0/1)", "1 = fraudulent claim (SM Verma), 0 = legitimate"),
        ])


# ═════════════════════════════════════════════════════════════
# CASE 3: BANK STRUCTURING
# ═════════════════════════════════════════════════════════════
elif page == "🏛️ Case 3: Bank Structuring":
    st.markdown("## 🏛️ Case Study 3: Bank Transaction Structuring (PMLA)")
    st.markdown(f"""<div class="hero-wrap" style="padding:20px 28px; text-align:left;">
      <span class="badge">Case Study 3</span>
      <span class="badge-red badge">AML / PMLA</span>
      <h3 style="color:{GOLD}; margin:10px 0 6px;">Cash Deposit Structuring to Evade ₹10 Lakh Reporting</h3>
      <p style="color:{TXT}; margin:0;"><b>Scenario:</b> A bank flags a customer who made 45 cash deposits over 60 days —
        deliberately structured to stay below the ₹10 lakh PMLA reporting threshold.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deposits", "45 transactions")
    col2.metric("Period", "60 days")
    col3.metric("Total Amount", "≈ ₹43 Lakhs")
    col4.metric("PMLA Threshold", "₹10 Lakhs")

    deposits = DF_CASE3["Amount_INR"].values
    st.session_state["case3_data"] = deposits
    result = benford_analysis(deposits)
    dates_series = pd.to_datetime(DF_CASE3["Date"])
    amounts_series = deposits

    tab1, tab2, tab3, tab4, tab_dl = st.tabs([
        "📋 PMLA Background", "📊 Benford Analysis",
        "⏱️ Time-Pattern Analysis", "📚 Learning Points", "📥 Download Raw Data"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""<div class="mp-card">
              <b style="color:{GOLD};">PMLA Reporting Obligations (India)</b><br><br>
              Banks file <b>CTR</b> for cash ≥ ₹10L in a single day. They file <b>STR to FIU-IND</b>
              when structuring is suspected. Structuring is itself a criminal offence (PMLA Sec 3).
              Penalty: imprisonment 3–7 years + attachment of funds.
            </div>
            <div class="mp-card-red">
              <b style="color:{RED};">Structuring ("Smurfing") Pattern</b><br><br>
              Multiple deposits each just below threshold · spread across multiple days ·
              sometimes multiple branches. Benford signature: <b>extreme digit-9 over-representation</b>
              (₹90k–₹99,999 when threshold is ₹1L).
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("**Deposit History (₹):**")
            st.dataframe(DF_CASE3[["Date","Amount_INR","Amount_Lakhs","Below_10L_Threshold"]].head(15),
                         use_container_width=True, hide_index=True)
            st.error("⚠️ Every single deposit is below ₹10 lakh — none trigger CTR individually!")

    with tab2:
        if result:
            show_stats_panel(result)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(chart_first_digit(result, "Cash Deposit Structuring Analysis"), use_container_width=True)
            with col2: st.plotly_chart(chart_deviation(result), use_container_width=True)
            show_digit_table(result)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(chart_log_histogram(deposits), use_container_width=True)
            with col2:
                fig_d = go.Figure(go.Histogram(x=deposits/100000, nbinsx=30, marker_color=RED, opacity=0.8))
                fig_d.add_vline(x=10, line_color=GOLD, line_width=2.5,
                    annotation_text="₹10L PMLA threshold", annotation_font_color=GOLD)
                fig_d.update_layout(**PL, title=dict(text="<b>All Deposits Cluster Just Below ₹10L</b>",
                    font=dict(color=RED, size=13)), height=320)
                st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(f"""<div class="verdict-bad">
              🚨 STRUCTURING CONFIRMED — DIGIT-9: {result['obs_p'][9]*100:.1f}% (Expected: 4.6%)
              | MAD = {result['mad']:.4f} | File STR with FIU-IND
            </div>""", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=dates_series, y=amounts_series/100000,
                mode="markers+lines", marker=dict(color=RED, size=8), line=dict(color=MID_BLUE, width=1.5)))
            fig_ts.add_hline(y=10, line_color=GOLD, line_width=2, line_dash="dash",
                annotation_text="₹10L threshold", annotation_font_color=GOLD)
            fig_ts.update_layout(**PL, title=dict(text="<b>Daily Cash Deposits Over 60 Days</b>",
                font=dict(color=GOLD, size=13)), height=320)
            st.plotly_chart(fig_ts, use_container_width=True)
        with col2:
            fig_cum = go.Figure(go.Scatter(x=dates_series, y=np.cumsum(amounts_series)/100000,
                fill="tozeroy", fillcolor="rgba(0,77,128,0.27)", line=dict(color=DARK_BLUE, width=2)))
            fig_cum.update_layout(**PL, title=dict(text="<b>Cumulative Deposits Over 60 Days</b>",
                font=dict(color=GOLD, size=13)), height=320)
            st.plotly_chart(fig_cum, use_container_width=True)
        total = sum(amounts_series)
        st.markdown(f"""<div class="mp-card-red">
          <b style="color:{RED};">Aggregate Picture Reveals the Scheme</b><br><br>
          45 deposits × average ₹{total/len(amounts_series)/100000:.1f}L = Total ₹{total/100000:.1f}L in 60 days.
          0 individual deposits triggered a CTR. Velocity: 0.75 deposits/day vs peer average of 2–3/month.
        </div>""", unsafe_allow_html=True)

    with tab4:
        for i,(title,content) in enumerate([
            ("Structuring = Digit-9 Signature","₹90k–₹9,99,999 → digit-9 spike reveals exact threshold gaming."),
            ("Aggregate Matters","No single CTR triggered, but ₹43L in 60 days is itself suspicious."),
            ("Benford + Velocity","Which digit + how often = complete investigator picture."),
            ("Legal Obligation: STR Filing","PMLA requires STR even when no single transaction crosses threshold."),
            ("FIU-IND Integration","Indian banks integrate transaction monitoring with FIU-IND for STR filing."),
        ], 1):
            st.markdown(f'<div class="mp-card-blue"><b style="color:{LIGHT_BLUE};">Learning {i}: {title}</b><br>{content}</div>', unsafe_allow_html=True)

    # ── DOWNLOAD TAB ────────────────────────────────────────
    with tab_dl:
        st.markdown(f"""<div class="mp-card">
        <b style="color:{GOLD};">📥 Case 3 Raw Dataset — Bank Transaction Structuring (PMLA)</b><br><br>
        Complete 45-transaction cash deposit dataset used in the structuring analysis.
        Includes transaction metadata, amounts, PMLA threshold flags, Benford digits,
        and cumulative running totals. Use this to practise digit analysis, velocity
        testing, and STR justification documentation.
        </div>""", unsafe_allow_html=True)

        dl_info(len(DF_CASE3), len(DF_CASE3.columns),
                "Simulated bank deposit data · Seed 404 · 60-day window · Jan–Feb 2024")

        st.dataframe(DF_CASE3, use_container_width=True, hide_index=True)

        if result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Digit-9 Frequency", f"{result['obs_p'][9]*100:.1f}%", delta="Expected 4.6%", delta_color="inverse")
            c2.metric("MAD", f"{result['mad']:.4f}", delta="Threshold 0.015", delta_color="inverse")
            c3.metric("Below ₹10L", f"{(DF_CASE3['Below_10L_Threshold']==1).sum()}/45", delta="All deposits", delta_color="off")
            c4.metric("CTR Triggered", "0", delta="Structuring confirmed", delta_color="off")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            dl_btn(DF_CASE3.to_csv(index=False).encode("utf-8"),
                   "case3_bank_structuring_pmla_data.csv",
                   "⬇️  Download Case 3 CSV — Bank Deposits (45 rows)")
        with col2:
            st.markdown(f"""<div class="mp-card-blue" style="padding:12px 16px;">
            <b style="color:{LIGHT_BLUE};">Suggested Exercises:</b><br>
            <span style="font-size:0.85rem;">
            1. Run benford_analysis() on Amount_INR — verify digit-9 Z-score<br>
            2. Plot amount distribution vs ₹10L threshold line<br>
            3. Compute velocity: deposits per week in this 60-day window<br>
            4. Draft an STR narrative using the statistical evidence from this dataset
            </span></div>""", unsafe_allow_html=True)

        col_dict_table([
            ("Transaction_ID",        "string",    "Sequential ID (TXN-001 to TXN-045)"),
            ("Date",                  "date",      "Deposit date — daily, Jan–Feb 2024"),
            ("Transaction_Type",      "string",    "Cash Deposit (all transactions)"),
            ("Amount_INR",            "float",     "Deposit amount in Indian Rupees (₹)"),
            ("Amount_Lakhs",          "float",     "Deposit amount in ₹ Lakhs (Amount_INR / 100,000)"),
            ("Below_10L_Threshold",   "int (0/1)", "1 if deposit < ₹10L (all are 1 in this dataset)"),
            ("First_Digit",           "int (1-9)", "Leading digit — extreme digit-9 over-representation"),
            ("Last_Digit",            "int (0-9)", "Last digit of deposit amount"),
            ("Benford_Expected_Pct",  "float",     "Benford's expected % for this first digit"),
            ("CTR_Triggered",         "int (0/1)", "Cash Transaction Report — 0 for all (structuring intent)"),
            ("Structuring_Flag",      "int (0/1)", "1 = this transaction is part of structuring scheme"),
            ("Cumulative_Amount_INR", "float",     "Running cumulative total — shows ₹43L aggregate"),
        ])

    # ── Cross-case combined download ───────────────────────
    st.markdown("---")
    st.markdown(f"""<div class="mp-card" style="margin-top:8px;">
    <b style="color:{GOLD};">📦 Download All Three Case Study Datasets</b><br>
    Get all three datasets in one ZIP archive for offline analysis, classroom exercises,
    or student distribution.
    </div>""", unsafe_allow_html=True)

    readme_txt = (
        "Mountain Path – World of Finance | Benford's Law Fraud Analytics\n"
        "Prof. V. Ravichandran | themountainpathacademy.com\n\n"
        "Files:\n"
        "  case1_gst_invoice_fraud_data.csv    -- 200 rows: GST invoice fraud (Seed 101)\n"
        "  case2_expense_fraud_data.csv         -- 1,250 rows: expense threshold gaming (Seed 202)\n"
        "  case3_bank_structuring_pmla_data.csv -- 45 rows: PMLA cash deposit structuring (Seed 404)\n\n"
        "All data is simulated and calibrated to realistic Indian market scenarios.\n"
        "Reference: Nigrini (2012) Benford's Law | ACFE 2024 Report to the Nations\n"
    )
    zip_buf = zip_three(
        DF_CASE1, "case1_gst_invoice_fraud_data.csv",
        DF_CASE2, "case2_expense_fraud_data.csv",
        DF_CASE3, "case3_bank_structuring_pmla_data.csv",
        readme=readme_txt,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        dl_btn(DF_CASE1.to_csv(index=False).encode("utf-8"),
               "case1_gst_invoice_fraud_data.csv", "⬇️  Case 1 CSV")
    with c2:
        dl_btn(DF_CASE2.to_csv(index=False).encode("utf-8"),
               "case2_expense_fraud_data.csv", "⬇️  Case 2 CSV")
    with c3:
        dl_btn(DF_CASE3.to_csv(index=False).encode("utf-8"),
               "case3_bank_structuring_pmla_data.csv", "⬇️  Case 3 CSV")

    st.download_button(
        label="📦  Download All Three Datasets — ZIP Archive (3 CSVs + README)",
        data=zip_buf,
        file_name="mountain_path_benford_case_studies.zip",
        mime="application/zip",
        use_container_width=True,
    )


# ═════════════════════════════════════════════════════════════
# PAGE: ML ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════
elif page == "🤖 ML Anomaly Detection":
    st.markdown("## 🤖 Machine Learning Anomaly Detection")
    tab1, tab2, tab3 = st.tabs(["🌲 Isolation Forest","📐 Multi-Method Framework","📚 ML Concepts"])

    with tab1:
        st.markdown(f"""<div class="mp-card-blue">
          <b style="color:{LIGHT_BLUE};">How Isolation Forest Works</b><br><br>
          Anomalous transactions are "easy to isolate" — they require very few random splits.
          Score ≈ 1 → clearly anomalous | ≈ 0.5 → normal | ≪ 0.5 → very normal
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            n_total = st.slider("Total transactions:", 1000, 5000, 3000, 100)
            contamination = st.slider("Expected fraud rate (%):", 1, 10, 3) / 100
            n_estimators = st.selectbox("Number of trees:", [100, 200, 300], index=1)
            inc_s = st.checkbox("Include structuring fraud", True)
            inc_v = st.checkbox("Include high-velocity fraud", True)
            inc_l = st.checkbox("Include large amount fraud", True)
            run_ml = st.button("🚀 Run Isolation Forest", type="primary")

        with col2:
            if run_ml:
                np.random.seed(42)
                n_clean = int(n_total * 0.95); n_fraud = n_total - n_clean
                ca = np.random.lognormal(9, 1.5, n_clean)
                ch = np.random.randint(8, 22, n_clean); cv = np.random.poisson(3, n_clean)
                cd = np.random.exponential(2, n_clean)
                fa, fh, fv, fd = [], [], [], []
                if inc_s:
                    nf=n_fraud//3; fa.extend(np.random.uniform(950000,999000,nf))
                    fh.extend(np.random.randint(8,22,nf)); fv.extend(np.random.poisson(3,nf))
                    fd.extend(np.random.exponential(2,nf))
                if inc_v:
                    nf=n_fraud//3; fa.extend(np.random.lognormal(9,1.5,nf))
                    fh.extend(np.random.randint(22,24,nf)); fv.extend(np.random.poisson(30,nf))
                    fd.extend(np.random.uniform(0,0.1,nf))
                if inc_l:
                    nf=n_fraud-len(fa)
                    if nf>0:
                        fa.extend(np.random.uniform(5000000,9999999,nf))
                        fh.extend(np.random.randint(8,22,nf)); fv.extend(np.random.poisson(5,nf))
                        fd.extend(np.random.exponential(1,nf))
                aa=np.concatenate([ca,fa]); ah=np.concatenate([ch,fh])
                av=np.concatenate([cv,fv]); ad=np.concatenate([cd,fd])
                tl=np.concatenate([np.zeros(n_clean), np.ones(len(fa))])
                def gfd(x):
                    try: return int(str(abs(int(x)))[0])
                    except: return 1
                be=np.array([BENFORD.get(gfd(x),0.1) for x in aa])
                df_ml=pd.DataFrame({"log_amount":np.log10(np.maximum(aa,1)),"hour":ah,
                    "velocity":av,"days_since_last":ad,"benford_expected":be})
                sc=StandardScaler(); Xs=sc.fit_transform(df_ml)
                iso=IsolationForest(n_estimators=n_estimators,contamination=contamination,random_state=42)
                ps=iso.fit_predict(Xs); ss=iso.score_samples(Xs)
                df_ml["predicted_fraud"]=(ps==-1).astype(int); df_ml["anomaly_score"]=ss
                df_ml["true_fraud"]=tl.astype(int); df_ml["amount"]=aa
                tp=int(((df_ml.predicted_fraud==1)&(df_ml.true_fraud==1)).sum())
                fp=int(((df_ml.predicted_fraud==1)&(df_ml.true_fraud==0)).sum())
                fn=int(((df_ml.predicted_fraud==0)&(df_ml.true_fraud==1)).sum())
                pr=tp/(tp+fp) if tp+fp>0 else 0; rc=tp/(tp+fn) if tp+fn>0 else 0
                f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0
                c1m,c2m,c3m,c4m=st.columns(4)
                c1m.metric("Precision",f"{pr*100:.1f}%"); c2m.metric("Recall",f"{rc*100:.1f}%")
                c3m.metric("F1-Score",f"{f1:.3f}"); c4m.metric("Flagged",f"{tp+fp:,}")
                ndf=df_ml[df_ml.predicted_fraud==0]; fdf=df_ml[df_ml.predicted_fraud==1]
                fig_sc=go.Figure()
                fig_sc.add_trace(go.Scatter(x=ndf.log_amount,y=ndf.velocity,mode="markers",
                    marker=dict(color=MID_BLUE,size=4,opacity=0.4),name="Normal"))
                fig_sc.add_trace(go.Scatter(x=fdf.log_amount,y=fdf.velocity,mode="markers",
                    marker=dict(color=RED,size=8,symbol="x",opacity=0.8),name="Flagged"))
                fig_sc.update_layout(**PL,height=380,title=dict(text="<b>Isolation Forest: Amount vs Velocity</b>",
                    font=dict(color=GOLD,size=13)),legend=dict(font=dict(color=TXT)))
                st.plotly_chart(fig_sc, use_container_width=True)

    with tab2:
        st.markdown(f"""<div class="mp-card">
          <b style="color:{GOLD};">Ensemble Anomaly Scoring</b><br>
          Score = w₁·S_Benford + w₂·S_IsoForest + w₃·S_Rules + w₄·S_Network
        </div>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Component":["Rule-Based","Benford's Law","Isolation Forest","Network / Velocity"],
            "Weight":["25%","20%","30%","25%"],
            "Detects Best":["Known patterns","Fabricated amounts","Multivariate anomalies","Collaborative fraud"]}),
            use_container_width=True, hide_index=True)
        for lbl, col, desc in [
            ("Layer 1: Rule-Based",DARK_BLUE,"Transaction limits • Threshold proximity • Duplicates"),
            ("Layer 2: Benford",MID_BLUE,"First-digit • MAD • Chi-squared • Z-score • Summation test"),
            ("Layer 3: ML","#6a0080","Isolation Forest • Autoencoder • DBSCAN • LOF"),
            ("Layer 4: Network","#008060","Graph analytics • Entity linkages • Velocity"),
            ("Layer 5: Human","#805000","Prioritised queue • Interview • Verification • Legal action"),
        ]:
            st.markdown(f"""<div style="background:{CARD_BG}; border-left:5px solid {col};
                border-radius:6px; padding:12px 16px; margin:6px 0;">
              <b style="color:{col};">{lbl}</b>
              <span style="color:{MUTED}; font-size:12px; margin-left:12px;">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with tab3:
        for title, content in [
            ("Supervised vs Unsupervised","Isolation Forest is unsupervised — learns 'normal' without fraud labels. Critical when fraud is rare and novel."),
            ("The Curse of Class Imbalance","Genuine fraud is 0.1–3% of transactions. Use Precision, Recall, F1 — not accuracy."),
            ("Precision vs Recall Trade-off","High Recall = fewer missed frauds. In fraud detection, Recall usually matters more."),
            ("Feature Engineering","Log-transform amounts, compute velocity, time-since-last, Benford probability as feature, rolling averages."),
            ("Model Drift","Re-train quarterly. Track Recall over time — declining Recall signals drift."),
        ]:
            with st.expander(f"💡 {title}", expanded=False):
                st.markdown(f'<div class="mp-card-blue"><span style="color:{TXT};">{content}</span></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: QUIZ
# ═════════════════════════════════════════════════════════════
elif page == "❓ Quiz & Assessment":
    st.markdown("## ❓ Knowledge Assessment — Benford's Law & Fraud Analytics")

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False

    questions = [
        {"q":"1. According to Benford's Law, what is the expected probability of a number beginning with digit 1?",
         "options":["11.1% (uniform)","30.1%","17.6%","25.0%"],"answer":1,
         "explain":"P(1) = log₁₀(2) ≈ 30.1%. Digit 1 appears as the leading digit in roughly 1 in 3 naturally occurring numbers."},
        {"q":"2. In a dataset of 1,000 expense claims, MAD = 0.022. Correct interpretation?",
         "options":["Close conformity","Acceptable conformity","Non-conforming — investigate immediately","Too few observations"],
         "answer":2,"explain":"MAD > 0.015 indicates non-conformity. Flag for investigation (Nigrini, 2012)."},
        {"q":"3. A bank customer makes 30 deposits of ₹9,50,000 over 90 days (PMLA threshold ₹10L). Which fraud?",
         "options":["Tax evasion through round numbers","Structuring (smurfing) under PMLA threshold","Financial statement manipulation","Expense fraud"],
         "answer":1,"explain":"Classic structuring (smurfing) — breaking a large sum into deposits below the ₹10L CTR threshold."},
        {"q":"4. Which dataset does NOT follow Benford's Law?",
         "options":["NSE daily trading volumes","GST invoice amounts","Employee ID numbers (sequential)","Financial statement line items"],
         "answer":2,"explain":"Sequential employee IDs are assigned uniformly — not generated by natural multiplicative processes."},
        {"q":"5. Chi-squared rejects at p=0.001 but MAD=0.008 (acceptable). What to conclude?",
         "options":["Fraud confirmed","Chi-squared wins — investigate","MAD is more reliable; deviation is statistically significant but practically small","Discard chi-squared"],
         "answer":2,"explain":"With very large samples, chi-squared oversensitively rejects Benford. MAD is the preferred metric."},
        {"q":"6. Digit 4 has Z-score 5.2 on 200 invoices, approval threshold ₹5,000. Best fraud hypothesis?",
         "options":["Psychological anchoring (₹4,999)","Threshold gaming — amounts just below ₹5,000","Rounding errors","Dataset too small"],
         "answer":1,"explain":"Digit-4 over-representation at ₹5k threshold is the classic threshold-gaming signature."},
        {"q":"7. A fraudster deliberately fabricates invoices conforming to Benford's first-digit distribution. Implication?",
         "options":["Benford is useless","Supplement with second digit, summation, last digit, ML tests","Fraudster cannot be caught","Use larger sample"],
         "answer":1,"explain":"A knowledgeable fraudster can defeat first-digit conformity but rarely all tests simultaneously."},
        {"q":"8. Required action under Indian law when bank detects PMLA structuring?",
         "options":["Freeze account immediately","Inform customer, 30 days to explain","File STR with FIU-IND","Wait for ₹10L to be crossed"],
         "answer":2,"explain":"PMLA and RBI Master Directions require STR with FIU-IND when structuring is suspected — regardless of individual transaction size."},
    ]

    if not st.session_state.quiz_submitted:
        st.markdown(f"""<div class="mp-card">
          <b style="color:{GOLD};">📋 Instructions</b><br>
          Answer all 8 questions, then click Submit. Each question carries 1 mark.
        </div>""", unsafe_allow_html=True)
        for i, q_data in enumerate(questions):
            sel = st.radio(q_data["q"], q_data["options"], key=f"q{i}", index=None)
            if sel: st.session_state.quiz_answers[i] = q_data["options"].index(sel)
        if st.button("✅ Submit Assessment", type="primary"):
            if len(st.session_state.quiz_answers) < len(questions):
                st.warning(f"Please answer all questions. ({len(st.session_state.quiz_answers)}/{len(questions)} answered)")
            else:
                score = sum(1 for i,q in enumerate(questions)
                    if st.session_state.quiz_answers.get(i)==q["answer"])
                st.session_state.quiz_score = score; st.session_state.quiz_submitted = True; st.rerun()
    else:
        score = st.session_state.quiz_score; pct = score/len(questions)*100
        if pct>=75: grade,css = f"Excellent! {score}/{len(questions)} ({pct:.0f}%)", "verdict-ok"
        elif pct>=50: grade,css = f"Good effort! {score}/{len(questions)} ({pct:.0f}%)", "verdict-warn"
        else: grade,css = f"Needs Review: {score}/{len(questions)} ({pct:.0f}%)", "verdict-bad"
        st.markdown(f'<div class="{css}">🎓 Your Score: {grade}</div>', unsafe_allow_html=True)
        st.progress(score/len(questions))
        st.markdown("### 📖 Detailed Explanations")
        for i, q_data in enumerate(questions):
            ua = st.session_state.quiz_answers.get(i,-1); correct = q_data["answer"]
            ok = ua==correct
            with st.expander(f"{'✅' if ok else '❌'} Q{i+1}: {q_data['q'][:70]}...", expanded=not ok):
                st.markdown(f"**Your answer:** {q_data['options'][ua] if ua>=0 else 'Not answered'}")
                st.markdown(f"**Correct answer:** {q_data['options'][correct]}")
                if ok: st.success(f"✅ Correct! {q_data['explain']}")
                else:  st.error(f"❌ Explanation: {q_data['explain']}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Quiz"):
                st.session_state.quiz_score=0; st.session_state.quiz_answers={}
                st.session_state.quiz_submitted=False; st.rerun()
        with col2:
            st.markdown(f"""<div class="mp-card">
              <b style="color:{GOLD};">📚 Continue Learning</b><br>
              Visit <a href="https://themountainpathacademy.com">themountainpathacademy.com</a>
              for full notes, Python notebooks, and extended case studies.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div style="text-align:center; color:{MUTED}; font-size:12px; padding:10px;">
      <b style="color:{GOLD};">The Mountain Path — World of Finance</b><br>
      Prof. V. Ravichandran |
      <a href="https://themountainpathacademy.com">themountainpathacademy.com</a> |
      <a href="https://www.linkedin.com/in/trichyravis">LinkedIn</a> |
      <a href="https://github.com/trichyravis">GitHub</a><br>
      <span style="font-size:10px;">Reference: Nigrini (2012) Benford's Law | ACFE 2024 Report to the Nations</span>
    </div>""", unsafe_allow_html=True)
