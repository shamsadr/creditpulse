# CreditPulse 📊
### Project 1 of 3 — Credit Risk Series

> *A live macroeconomic credit risk dashboard powered by real Federal Reserve data.
> Pulls, cleans, and visualizes 6 key credit indicators to answer one question:
> **How risky is it to lend money right now, compared to history?***

---

## Live Demo

🚀 **[Launch Dashboard →](https://credit-pulse.streamlit.app/)** *(update link after deploying)*

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Data](https://img.shields.io/badge/Data-FRED%20API-red)

---

## Problem

Credit risk analysts and quant teams rely on macroeconomic signals — delinquency rates,
credit spreads, unemployment — to forecast loan defaults and stress-test portfolios.
This dashboard makes those signals visible and interpretable in real time,
directly from the Federal Reserve's data API.

## The Core Insight

Credit risk follows a predictable sequence:

```
Credit spreads widen → Unemployment rises → Delinquencies spike → Charge-offs follow
```

**Credit spreads lead by 3–6 months.** Delinquencies confirm what spreads already predicted.
The Credit Stress Index combines both into a single read on the current credit cycle.

## What the Dashboard Shows

| Section | What it answers |
|---------|----------------|
| **Bottom Line** | Current stress level vs. history (percentile rank since 2000) |
| **Credit Stress Index** | Composite leading/lagging indicator with recession shading |
| **6 FRED Indicators** | Individual time series: delinquency, charge-offs, spreads, unemployment, fed funds |
| **Correlation Heatmap** | Which indicators move together — and which lead vs. lag |
| **Data Quality Report** | Transparency on missing data, pull timestamps, series provenance |

## Key Indicators

| Series | FRED ID | Type |
|--------|---------|------|
| Credit card delinquency rate | `DRCCLACBS` | Lagging |
| Commercial loan delinquency | `DRBLACBS` | Lagging |
| Credit card charge-off rate | `CORCCACBS` | Lagging |
| Unemployment rate | `UNRATE` | Coincident |
| BAA credit spread | `BAA10Y` | **Leading** |
| Federal funds rate | `FEDFUNDS` | Policy |

## Tech Stack

- **Python 3.11+** — pandas, numpy, plotly, streamlit, fredapi
- **Data:** Federal Reserve Bank of St. Louis (FRED) — free, authoritative, reproducible
- **Pipeline:** modular ETL — fetch → validate → engineer → visualize
- **Features engineered:** lags (1/3/6 month), rolling means, MoM changes, credit stress index

## How to Run Locally

**Step 1 — Get a free FRED API key (30 seconds):**
```
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key" — instant approval
```

**Step 2 — Clone and install:**
```bash
git clone https://github.com/shamsadr/creditpulse.git
cd creditpulse
pip install fredapi pandas numpy plotly streamlit
```

**Step 3 — Run the dashboard:**
```bash
export FRED_API_KEY="your_key_here"
streamlit run app.py
```

**Step 4 — Or run the CLI pipeline only:**
```bash
python src/main.py
# Output: data/creditpulse_features.csv + data/metadata.json
```

**Step 5 — Run tests:**
```bash
pytest tests/ -v
```

## Series Architecture

```
credit-risk-series/
├── 01-creditpulse/        ← YOU ARE HERE: live data pipeline + dashboard
├── 02-creditscore/        ← Coming next: PD model (logistic regression scorecard)
└── 03-creditmigration/    ← Coming later: Markov chain credit rating transition model
```

`creditpulse_features.csv` feeds directly into Project 2 as the modeling dataset.

---

## Interview Talking Points

- **What problem are you solving?**
  "Credit risk models are only as good as their inputs — I built the data foundation first,
  pulling real Federal Reserve indicators and engineering features that proxy for credit
  cycle stress, then visualized them in a live dashboard."

- **Why this approach?**
  "FRED is the authoritative source for U.S. macro data, it's free, and fredapi gives
  reproducible pulls — any analyst can verify my data provenance, which matters in a
  risk context."

- **What does the dashboard tell you right now?**
  "Credit stress is at the 15th percentile since 2000 — conditions are relatively benign.
  Spreads are tight and delinquencies are moderate, which historically precedes either
  continued stability or a gradual turn depending on where the Fed moves rates."

- **What would you improve with more time?**
  "I'd add loan-level data (HMDA or Fannie Mae performance data) and automate daily
  refreshes so the dashboard stays current without manual reruns."

- **What did you learn?**
  "Credit spreads lead delinquencies by 3–6 months — the lag structure in the data
  confirms what theory predicts, and it's directly visible in the correlation heatmap."

---

*Part of a 3-project credit risk portfolio.*
*See also: CreditScore (Project 2) and CreditMigration (Project 3) — coming soon.*

---

*Built by [Shamsad Rahman](https://github.com/shamsadr) · Data: [FRED](https://fred.stlouisfed.org)*
