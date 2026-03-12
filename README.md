## Baseball Season Win Predictor

This project predicts MLB team regular-season win totals using historical team batting and pitching statistics, Vegas win totals, and engineered features such as Pythagorean expectation and lagged metrics.

### Key files

- **Baseball Predictor.py**: End-to-end script that:
  - Pulls team batting and pitching data from `pybaseball`
  - Merges in Vegas win totals and player stats
  - Engineers features (Pythagorean wins, WAR-based metrics, lagged features)
  - Trains an ensemble model (`XGBRegressor`, `RandomForestRegressor`, and `Ridge` in a `VotingRegressor`)
  - Evaluates performance on recent seasons and saves a feature-importance chart


### Setup

1. Create and activate a Python environment (Python 3.9+ recommended).
2. Install dependencies, for example:

```bash
pip install pandas numpy pybaseball xgboost scikit-learn matplotlib
```

3. Make sure the CSV files used by the script are present in the project directory:
   - `vegas_wins.csv`
   - `player_stats.csv`
   - (Optional) any additional data sources you add later

### Usage

From the project directory, run:

```bash
python "Baseball Predictor.py"
```

The script will:
- Fetch and cache data from `pybaseball`
- Train the ensemble model
- Print RMSE/MAE and per-team predictions for the most recent seasons
- Save `feature_importance.png` showing averaged feature importance

