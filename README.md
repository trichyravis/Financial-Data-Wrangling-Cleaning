# 🏔️ Benford's Law — Fraud Analytics & Anomaly Detection
### The Mountain Path — World of Finance
**Prof. V. Ravichandran | [themountainpathacademy.com](https://themountainpathacademy.com)**

---

## 📋 Overview

A comprehensive, interactive Streamlit learning platform covering:

| Module | Content |
|--------|---------|
| 📖 **Learn** | History, mathematics, intuition, statistical tests, ethics |
| 📊 **Analyzer** | Upload your own data or generate synthetic datasets |
| 🏦 **Case 1** | GST Invoice Fraud — fictitious ITC claims (₹5 Cr fraud) |
| 💼 **Case 2** | Expense Report Fraud — threshold gaming by sales manager |
| 🏛️ **Case 3** | Bank Structuring — PMLA cash deposit smurfing |
| 🤖 **ML** | Isolation Forest multi-feature anomaly detection |
| ❓ **Quiz** | 8-question knowledge assessment with detailed explanations |

---

## 🚀 Quick Start

### Local Installation
```bash
# Clone / copy files to a directory
cd benford_app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Google Colab
```python
!pip install streamlit pyngrok -q
!pip install -r requirements.txt -q

# Run with tunnel
from pyngrok import ngrok
import subprocess
proc = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', '8501'])
public_url = ngrok.connect(8501)
print(f"App URL: {public_url}")
```

### Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → deploy

---

## 📐 Benford's Law Formula

```
P(d) = log₁₀(1 + 1/d)    for d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
```

| Digit | Probability | Meaning |
|-------|-------------|---------|
| 1 | **30.10%** | Appears in ~1 in 3 natural numbers |
| 2 | 17.61% | About 1 in 6 |
| 3 | 12.49% | About 1 in 8 |
| 9 | 4.58% | Appears only ~1 in 22 |

---

## 🔍 Three Case Studies

### Case 1: GST Invoice Fraud (₹5 Crore ITC Scam)
- Textile trader claiming fictitious Input Tax Credit
- 200 fabricated invoices from shell companies
- Benford signals: Digit-9 spike (30% vs expected 4.6%), MAD >> 0.015
- Outcome: ₹5 Cr demand + ₹10 Cr penalty + criminal prosecution

### Case 2: Expense Report Fraud (Corporate)
- Sales manager submitting 250 claims vs peer average of 38
- Threshold gaming: 60%+ of claims in ₹4,200–₹4,999 range (below ₹5K limit)
- Per-employee Benford testing reveals the anomaly
- Round number test: 40% of last digits are 0 or 5

### Case 3: Bank Structuring (PMLA Anti-Money Laundering)
- 45 cash deposits of ₹9.2–9.98 lakh each (all below ₹10L CTR threshold)
- Digit-9 over-representation: 78% vs expected 4.6%
- Velocity analysis: 0.75 deposits/day vs peer 0.1/day
- PMLA Suspicious Transaction Report (STR) filing obligation triggered

---

## 📊 Statistical Tests Implemented

| Test | Formula | Use Case |
|------|---------|---------|
| **MAD** | (1/9)Σ\|P_obs - P_benford\| | Primary test; not sample-size inflated |
| **Chi-Squared** | Σ(O-E)²/E | Goodness-of-fit; df=8 |
| **Z-Score/digit** | Per-digit significance | Identify which digit is anomalous |
| **KS Test** | max\|F_obs - F_benford\| | Small samples; cumulative comparison |
| **Last-Digit** | Uniformity of last digits | Round-number / fabrication detection |
| **Summation** | Value concentration by 2-digit | Threshold gaming; high-value clusters |

---

## 🤖 Machine Learning Module

- **Isolation Forest** with configurable contamination rate
- Features: log(amount), hour, velocity, recency, Benford expected probability
- Performance metrics: Precision, Recall, F1-Score
- Visualisations: scatter (amount vs velocity), anomaly score distribution
- Ensemble scoring framework explanation

---

## 📚 Educational Content

- Plain-English explanations before mathematics
- Step-by-step compound growth intuition for Benford's Law
- Interactive digit probability explorer (slider)
- Complete statistical test reference with critical values
- Ethics framework: presumption of innocence, fairness, DPDP Act
- 8-question quiz with instant feedback and detailed explanations

---

## 🎨 Design

Built with The Mountain Path design system:
- **Colors:** `#003366` (dark blue), `#FFD700` (gold), `#ADD8E6` (light blue)
- **Background:** `linear-gradient(135deg, #1a2332, #243447, #2a3f5f)`
- **Charts:** Plotly with custom Mountain Path theme
- Fully responsive; works on desktop and tablet

---

## 📖 References

1. Nigrini, M.J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.
2. ACFE (2024). *Report to the Nations: 2024 Global Study on Occupational Fraud and Abuse*.
3. Liu, F.T. et al. (2008). Isolation Forest. *IEEE ICDM*.
4. RBI (2023). *Annual Report on Banking Fraud Trends*.

---

## 🔗 Links

- 🌐 [themountainpathacademy.com](https://themountainpathacademy.com)
- 💼 [LinkedIn: Prof. V. Ravichandran](https://www.linkedin.com/in/trichyravis)
- 💻 [GitHub: trichyravis](https://github.com/trichyravis)

---

*© 2025 The Mountain Path — World of Finance. For educational use.*
