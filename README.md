# CreditPulse 📊
### Project 1 of 3 — Credit Risk Series

> *A production-ready data pipeline for pulling, cleaning, and feature-engineering macroeconomic
> credit indicators from the Federal Reserve (FRED). Foundation for downstream PD modeling and
> credit migration analysis.*

---

## Problem

Credit risk analysts and quant teams rely on macroeconomic signals — delinquency rates, credit
spreads, unemployment — to forecast loan defaults and stress-test portfolios. Building a clean,
reproducible pipeline for these indicators is the unglamorous but essential first step that most
academic projects skip. This project does it right.

## Approach

- Pulls 6 key FRED credit indicators via the `fredapi` library (free, real data)
- Cleans and aligns time series to a common monthly frequency
- Engineers lag features, rolling statistics, and recession flags (NBER-based)
- Exports a modeling-ready dataset with full provenance metadata
- Designed to feed directly into Project 2 (PD scorecard) and Project 3 (Markov migration model)

**Key indicators pulled:**
| Series | FRED ID | What it measures |
|--------|---------|-----------------|
| Consumer credit delinquency rate | `DRCCLACBS` | % of credit card loans 30+ days past due |
| Commercial & industrial loan delinquency | `DRBLACBS` | Business loan stress |
| Credit card charge-off rate | `CORCCACBS` | Realized losses on credit cards |
| Unemployment rate | `UNRATE` | Macro stress indicator |
| BAA-AAA credit spread | `BAA10Y` | Market-implied credit risk premium |
| Federal funds rate | `FEDFUNDS` | Interest rate environment |

**Design choices:**
- Forward-fill then backward-fill for minor gaps (< 3 months); flag longer gaps
- All series normalized to 2000-01-01 baseline for comparability
- Recession periods flagged using FRED's `USREC` indicator

## Results / Key Outputs

Running `python src/main.py` produces:

1. `data/creditpulse_features.csv` — cleaned, feature-engineered dataset (monthly, 2000–present)
2. `data/metadata.json` — pull timestamp, series descriptions, missing data report
3. Console summary: date range, shape, missing value counts, correlation matrix snippet

## Tech Stack

- Python 3.11+, `fredapi`, `pandas`, `numpy`, `requests`
- Modular ETL design: fetch → validate → engineer → export (each step independently testable)
- FRED API key via environment variable (never hardcoded)

## How to Run

**Step 1 — Get a free FRED API key (30 seconds):**
```
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key" — instant approval
3. Copy your key
```

**Step 2 — Set up environment:**
```bash
conda env create -f environment.yml
conda activate creditpulse
```

**Step 3 — Add your API key:**
```bash
export FRED_API_KEY="your_key_here"
# Or add to ~/.zshrc to persist it
```

**Step 4 — Run:**
```bash
python src/main.py
# With custom date range:
python src/main.py --start 2010-01-01 --end 2024-12-31
# Offline mode (uses cached data if available):
python src/main.py --offline
```

**Step 5 — Run tests:**
```bash
pytest tests/ -v
```

## Series Architecture

```
credit-risk-series/
├── 01-creditpulse/        ← YOU ARE HERE: data pipeline
├── 02-creditscore/        ← Coming next: PD model (logistic regression scorecard)
└── 03-creditmigration/    ← Coming later: Markov chain rating transition model
```

Each project imports from the previous one. `creditpulse_features.csv` is the input to Project 2.

---

## Interview Talking Points

- **What problem are you solving?**
  "Credit risk models are only as good as their inputs — I built the data foundation first, pulling
  real Federal Reserve indicators and engineering features that proxy for credit cycle stress."

- **Why this approach?**
  "FRED is the authoritative source for U.S. macro data, it's free, and fredapi gives reproducible
  pulls — any analyst can verify my data provenance, which matters in a risk context."

- **What would you improve with more time?**
  "I'd add loan-level data (e.g., HMDA mortgage data or Fannie Mae loan performance) and build
  an automated refresh scheduler so the pipeline stays current."

- **What did you learn?**
  "Macro credit indicators are highly autocorrelated and lag the business cycle — the feature
  engineering (lags, rolling stats, recession flags) is where the real signal lives."

---

*Part of a 3-project credit risk portfolio. See also: [CreditScore](../02-creditscore) and [CreditMigration](../03-creditmigration)*
