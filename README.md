<p align="center">
  <img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/XGBoost-Engine-FF6600?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MongoDB-Database-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
</p>

<h1 align="center">🏀 NBA Trade Analyzer</h1>
<p align="center">
  <b>AI-Powered Trade Evaluation · Player Forecasting · Injury Intelligence · Win Projections</b>
</p>
<p align="center">
  <i>Stop guessing. Start analyzing. Make every trade decision backed by data.</i>
</p>

<p align="center">
  <a href="#-how-to-use-the-website"><b>How to Use</b></a> •
  <a href="#-how-it-works"><b>How It Works</b></a> •
  <a href="#-features"><b>Features</b></a> •
  <a href="#-models--intelligence-layer"><b>ML Architecture</b></a>
</p>

---

## 🔗 Live Demo
> **Live Website:** https://nba-trade-analyser.vercel.app/

---

## 📖 How to Use the Website

Follow these steps to perform your first AI-backed trade analysis:

1. **Select Teams:** Choose two NBA franchises from the dropdown menus.
2. **Select Players:** Once the rosters load, click on players to add them to the trade block. The system handles "Outgoing" and "Incoming" logic automatically.
3. **Analyze:** Hit the **"Analyze Trade"** button. The system will run 1,000 Monte Carlo simulations in the background.
4. **Review the Results:**
   - **Trade Score:** Check the 0–100 gauge to see if the trade is beneficial.
   - **Win Delta:** See how many extra wins (or losses) the model projects for each team.
   - **"Why this prediction?":** Click on any traded player to see the SHAP-powered plain-English explanations of their forecast.
   - **Fit & Health:** Review the roster depth and injury risk warnings at the bottom of the analysis.

---

## What is NBA Trade Analyzer?

NBA Trade Analyzer is a full-stack intelligent web application that **predicts how NBA players will perform after a trade** — not just how they performed last season. It combines machine learning, injury analytics, salary cap rules, and Monte Carlo simulations to produce a single, definitive **Trade Score (0–100)** that tells you whether a deal helps or hurts your team.

Every prediction comes with a **plain-English explanation** powered by Explainable AI, so you always know *why* the system made its judgment.

---

## How It Works

You select two NBA teams. You pick the players each team is sending out. You hit **Analyze Trade**. In under 3 seconds, the system:

1. **Predicts next-season performance** for every player involved using 5 years of historical patterns.
2. **Classifies injury risk** by analyzing each player's medical history and games missed.
3. **Checks roster fit** — detects positional logjams and scoring cannibalization that would limit production.
4. **Simulates 500 possible seasons** using Monte Carlo methods to project total team wins before and after the trade.
5. **Validates salary cap compliance** to ensure the trade is financially feasible under NBA CBA rules.
6. **Calculates a final Trade Score** with a full breakdown of what's driving the number up or down.

---

## Features

- **Multi-Stat Forecasting** — Simultaneously predicts Points, Rebounds, Assists, Minutes, and True Shooting % for the upcoming season.
- **Explainable AI** — Every prediction includes human-readable SHAP explanations. No black boxes.
- **Injury Intelligence** — Medical risk grades from Low to Very High, factored directly into win projections.
- **Monte Carlo Win Simulator** — 500-iteration probabilistic engine projecting how many games a team wins post-trade.
- **Salary Cap Engine** — Financial feasibility checks against NBA salary matching rules.
- **Fit Penalty System** — Automatically penalizes trades that create positional overlap or scoring redundancy.
- **Confidence Intervals** — Every stat prediction includes upper/lower bounds based on model accuracy margins.
- **Fully Responsive UI** — Premium dark-mode Glassmorphism interface that works on desktop, tablet, and mobile.
- **Cinematic Intro** — Gate-opening animation with basketball iconography on every page load.

---

## Models & Intelligence Layer

The system runs **three distinct machine learning models** working in concert, plus a rule-based simulation engine.

### 1. Player Performance Model — XGBoost MultiOutput Regressor

The core prediction engine. It does not simply look at last season's averages — it processes **54 engineered temporal features** across a 5-year rolling window per player.

**Feature Categories:**

| Category | Features | What They Capture |
| :--- | :---: | :--- |
| Base Statistics | 16 | Current season metrics: PPG, RPG, APG, TS%, usage rate, points per minute, etc. |
| Lag Features | 26 | Historical stats going back 1–5 seasons (e.g., PPG from 3 years ago vs today) |
| Trend Slopes | 4 | Mathematical trajectory — is the player improving or declining year-over-year? |
| Career Architecture | 8 | Peak PPG, years since peak, career consistency (coefficient of variation), career averages |

**Targets Predicted:**
| Metric | Description |
| :--- | :--- |
| PPG | Points Per Game |
| RPG | Rebounds Per Game |
| APG | Assists Per Game |
| MPG | Minutes Per Game |
| TS% | True Shooting Percentage |

**Validated Accuracy (Backtested vs. actual 2024–25 season):**

| Metric | MAE | R² | What This Means |
| :--- | :---: | :---: | :--- |
| PPG | 1.37 | 0.919 | Predictions land within ~1 basket of actual scoring output |
| RPG | 0.52 | 0.908 | Highly reliable rebounding forecasts |
| APG | 0.36 | 0.926 | Excellent assist prediction precision |
| MPG | 2.81 | 0.847 | Strong minutes projection with minor variance |
| TS% | 0.04 | -1.21 | Known limitation — shooting % is inherently volatile season-to-season |

**Hyperparameters:** `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `min_child_weight=5`

### 2. Injury Risk Classifier — Random Forest

Evaluates 5 years of player medical history — total games missed, severity classifications (minor/moderate/severe), and frequency patterns — to assign a probabilistic injury risk rating.

| Risk Category | Probability Range | Impact on Trade Score |
| :--- | :---: | :--- |
| Low | < 15% | Minimal penalty |
| Moderate | 15–35% | Moderate reduction in projected wins |
| High | 35–60% | Significant penalty; flagged in UI |
| Very High | > 60% | Severe reduction; trade likely rated "Harmful" |

**Validation:** ROC-AUC of 0.575 against held-out 2024–25 injury data. Reflective of the inherent randomness of sports injuries — the model identifies patterns where they exist without overfitting to noise.

### 3. SHAP Explainability Engine — TreeExplainer

Every single prediction is run through a SHAP TreeExplainer that decomposes the XGBoost output into individual feature contributions. These raw SHAP values are then translated into natural-language sentences displayed directly in the UI.

**Example outputs:**
- *"His 3-year scoring trend is strongly upward, adding +2.1 points to the projection."*
- *"At age 35, the age factor is pulling the estimate down by -1.8 points."*
- *"Career consistency is very high (low variance), giving the model high confidence."*

### 4. Monte Carlo Win Simulation Engine

For every trade evaluation, the system simulates **500 complete NBA seasons** probabilistically. Each iteration:
- Injects Gaussian noise based on model RMSE to reflect real-world variance.
- Applies positional fit penalties (scoring cannibalization, roster depth issues).
- Factors in injury probability as a games-missed multiplier.
- Computes the team's expected win total for that simulated season.

The final projected win delta is the **mean across all 500 simulations**, producing a statistically robust estimate rather than a single fragile prediction.

### 5. Trade Score Formula

All model outputs are synthesized into a single **0–100 Trade Score** via a weighted composite:

| Component | Weight | Source |
| :--- | :---: | :--- |
| Win Delta | 50% | Monte Carlo simulation engine |
| Playoff Probability Change | 20% | Sigmoid mapping of projected wins to playoff odds |
| Injury Risk Differential | 15% | Random Forest classifier |
| Roster Health Impact | 10% | Aggregate team medical depth score |
| Individual Medical Score | 5% | Traded player medical grade differential |

**Score Interpretation:**

| Score Range | Rating | Meaning |
| :---: | :--- | :--- |
| 80–100 | Exceptional | Franchise-altering improvement |
| 65–79 | Beneficial | Clear net positive for the team |
| 50–64 | Neutral | Trade is roughly a wash |
| 35–49 | Risky | Likely hurts the team's outlook |
| 0–34 | Harmful | Significant downgrade projected |

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React 18, Vite, Tailwind CSS v3, Framer Motion, GSAP, Recharts, Lucide Icons |
| **Backend** | Python 3.12, Flask 3.0, Pandas, NumPy |
| **ML / AI** | XGBoost, Scikit-learn, SHAP |
| **Database** | MongoDB (pymongo) |
| **Data** | 5 seasons of NBA box scores, injury logs, team ratings, and salary data (2020–2025) |

---

## API Reference

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/api/health` | Returns system health, model load status, and player count |
| `GET` | `/api/teams` | Returns all 30 NBA franchise codes and names |
| `GET` | `/api/players?team={code}` | Returns the active roster for a given team abbreviation |
| `POST` | `/api/predict` | Returns predicted stats, confidence ranges, and SHAP explanations for a single player |
| `POST` | `/api/trade/evaluate` | Full trade evaluation — stats, injury risk, Monte Carlo wins, fit penalties, and Trade Score |
| `GET` | `/api/model/info` | Returns model metadata including feature list, target names, and per-target accuracy metrics |

---

## License

For academic and research use.