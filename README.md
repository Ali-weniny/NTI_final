# ⚽ Player Market Value Predictor

A machine learning project that predicts a football player's market value (`value_eur`) based on their technical attributes, age, position, and other real-world factors. Built as part of the ML Capstone project using the FC26 player ratings dataset.

## 📌 Problem Statement

Estimating a football player's market value is a real challenge faced by clubs, agents, and scouts during transfer negotiations. This project builds a two-stage machine learning pipeline that:

1. Predicts a player's **Overall rating** from their detailed technical stats (finishing, passing, dribbling, etc.)
2. Uses that predicted Overall rating — along with age, potential, league, contract details, and other features — to predict the player's **Market Value (€)**

This mirrors how a scout might reason: first judge the player's technical quality, then translate that into an expected transfer value.

## 📊 Dataset

- **Source:** FC26 player ratings dataset (`FC26_20250921.csv`)
- **Original size:** 18,405 players × 110 columns
- **After cleaning:** 16,266 players × 54 columns
- Goalkeepers were excluded from this project, since they are rated on a completely different set of attributes (goalkeeping stats) than outfield players.

## 🛠️ Project Structure

```
final-project/
├── Data/
│   ├── raw/
│   │   └── FC26_20250921.csv
│   └── processed/
│       └── clean_data.csv
│
├── notebooks/
│   ├── 01_cleaning_preprocessing.ipynb   # Data cleaning + feature engineering
│   ├── 02_eda.ipynb                      # Exploratory data analysis & visualizations
│   └── 03_model_training.ipynb           # Model training & evaluation
│
├── models/
│   ├── overall_model.pkl                 # Trained model: predicts Overall rating
│   ├── overall_features.pkl              # Feature list for the Overall model
│   ├── value_model.pkl                   # Trained model: predicts Market Value
│   └── value_features.pkl                # Feature list for the Value model
│
├── app/
│   └── app.py                            # Streamlit web application
│
├── requirements.txt
└── README.md
```

## 🧹 Data Cleaning & Preprocessing

- Removed columns with no predictive value or heavy missing data (`work_rate`, `player_tags`, `player_traits`, `nation_position`, etc.)
- Removed `release_clause_eur` to avoid data leakage (it is derived from `value_eur` in-game)
- Removed goalkeepers and all goalkeeping-specific columns
- Removed players without a current club (free agents)
- **Feature engineering:**
  - `years_at_club` — derived from `club_joined_date`
  - `contract_years_remaining` — derived from `club_contract_valid_until_year`
  - `is_loaned` — binary flag from `club_loaned_from`
  - `position_group` — simplified `player_positions` into DEF / MID / ATT, then one-hot encoded
  - `nationality_freq`, `league_freq` — frequency encoding for high-cardinality categorical columns (158 nationalities, 42 leagues) instead of one-hot encoding
  - `preferred_foot` — one-hot encoded (low cardinality)

## 📈 Exploratory Data Analysis

Key findings (see `02_eda.ipynb` for full visualizations):
- `value_eur` is heavily right-skewed → a `log1p` transform was applied before model training
- `overall` and `potential` are the features most strongly correlated with market value
- The relationship between `age` and `value_eur` is non-linear (value rises then falls with age)
- Attacking players (`ATT`) have a higher median market value than midfielders and defenders

## 🤖 Modeling Approach

### Stage 1 — Overall Rating Model
- **Input:** 33 detailed technical stats (finishing, dribbling, vision, composure, etc.)
- **Target:** `overall`
- **Model:** Random Forest Regressor
- **Result:** R² = 0.974, RMSE = 1.11

> Note: this very high accuracy is expected — a player's Overall rating is itself derived from these same sub-stats inside the game's own formula, not an independent value. This confirms that Stage 2 (predicting market value) is the real modeling challenge, since it is not a directly computed formula.

### Stage 2 — Market Value Model
- **Input:** predicted Overall (from Stage 1) + potential, age, league level, contract details, reputation, position group, nationality/league frequency, etc. (`wage_eur` excluded to avoid leakage)
- **Target:** `log1p(value_eur)`
- **Models compared:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.624 | 0.251 | 0.798 |
| **Random Forest** | **0.308** | **0.111** | **0.951** |

Random Forest was selected as the final model — the large improvement over the linear baseline shows that the relationship between player attributes and market value is non-linear and involves feature interactions.

## 💻 Application

A Streamlit web app (`app/app.py`) lets a user enter a player's technical stats and profile information, then:
1. Predicts the player's Overall rating (Stage 1)
2. Feeds that prediction into Stage 2 to estimate the player's market value in €

### Running the app
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

> **Note on environment:** This project was developed and tested with **Python 3.11**. Some dependencies (scikit-learn, xgboost) may not yet provide prebuilt packages for very new Python versions (e.g. 3.14) on Windows, which can cause installation errors. If you hit build errors during `pip install`, install Python 3.11 and create the virtual environment with it (`py -3.11 -m venv .venv`).

## 🔧 Requirements

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
xgboost==2.1.0
matplotlib==3.9.1
seaborn==0.13.2
streamlit==1.37.0
joblib==1.4.2
```

## 👥 Team

| Member | Responsibility |
|---|---|
| [Name] | Data Cleaning & Preprocessing |
| [Name] | Regression Modeling (Market Value) |
| [Name] | Overall Rating Model & Feature Engineering |
| [Name] | Streamlit App & Integration |
| [Name] | Documentation & Presentation |

## 📎 Notes

- The dataset only contains a single FC26 snapshot (one row per player), so no player growth/career trajectory modeling was possible with this data alone.
- Goalkeepers are not supported by the current models, since they were excluded during cleaning.
