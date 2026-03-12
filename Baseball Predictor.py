import pandas as pd
import numpy as np
import pybaseball.cache
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pybaseball
import time
import os



# Retrieve Data
pybaseball.cache.enable()
print("Fetching batting data...")
batting1 = pybaseball.team_batting(2004, 2014)
time.sleep(5)
batting2 = pybaseball.team_batting(2015, 2025)
time.sleep(5)
batting = pd.concat([batting1, batting2]).reset_index(drop=True)

print("Fetching pitching data...")
pitching1 = pybaseball.team_pitching(2004, 2014)
time.sleep(5)
pitching2 = pybaseball.team_pitching(2015, 2025)
time.sleep(5)
pitching = pd.concat([pitching1, pitching2]).reset_index(drop=True)

wins_df = pitching[['Season', 'Team', 'W']].rename(columns={'W': 'actual_wins'})


print("Reading CSVs...")
vegas = pd.read_csv('vegas_wins.csv')
player_stats = pd.read_csv('player_stats.csv')

# Fix Data
vegas['Team'] = vegas['Team'].replace({'Cleveland': 'CLE'})
batting['Team'] = batting['Team'].replace({'ATH': 'OAK'})
pitching['Team'] = pitching['Team'].replace({'ATH': 'OAK'})

print(sorted(vegas['Season'].unique()))

# Merge and Feature enginering
df = pd.merge(batting, pitching, on=['Season', 'Team'], suffixes=('_bat', '_pit'))
df = pd.merge(df, wins_df, on=['Season', 'Team'])
df = pd.merge(df, vegas, on=['Season', 'Team'])
df = pd.merge(df, player_stats, on = ['Season', 'Team'], how = 'left')


df = df[df['Season'] != 2020]

df['pyth_win_pct'] = df['R_bat'] ** 1.83 / (df['R_bat'] ** 1.83 + df['R_pit'] ** 1.83)
df['pyth_wins'] = df['pyth_win_pct'] * 162
df['pyth_diff'] = df['actual_wins'] - df['pyth_wins']

df = df.sort_values(['Team', 'Season']).reset_index(drop=True)

feature_cols = [
    'pyth_wins',    # best prior-year team quality measure
    'xFIP',         # best pitching predictor
    'wOBA',         # best offensive predictor
    'total_bat_WAR',
    'total_pit_WAR',
    'weighted_xFIP',
    'weighted_wOBA',
]

#Lags all Features by 1 season
for col in feature_cols:
    df[f'{col}_prev'] = df.groupby('Team')[col].shift(1)
    
df['wins_prev'] = df.groupby('Team')['actual_wins'].shift(1)
df['target_wins'] = df['actual_wins']

df = df.dropna(subset=[f'{col}_prev' for col in feature_cols] + ['wins_prev'])



# Model Training
# Use vegas_wins (current season odds) + lagged performance metrics
X_cols = [f'{col}_prev' for col in feature_cols] + ['wins_prev', 'vegas_wins']
X = df[X_cols]
Y = df['target_wins']

train_mask = df['Season'] <= 2023
test_mask = df['Season'] >= 2024

X_train, Y_train = X[train_mask], Y[train_mask]
X_test, Y_test = X[test_mask], Y[test_mask]


# Define individual models
xgb = XGBRegressor(n_estimators=466, learning_rate=0.07248594659468281, max_depth=2, subsample = 0.7126187570006727 , colsample_bytree = 0.6101480960613607, min_child_weight = 3, random_state=42)
rf = RandomForestRegressor(n_estimators=304, max_depth=6, min_samples_leaf = 2, random_state=42)
# Added StandardScaler to Ridge Pipeline
ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=11.004516679781556))

# Ensemble
ensemble = VotingRegressor(estimators=[
    ('xgb', xgb),
    ('rf', rf),
    ('ridge', ridge_pipe)
])

ensemble.fit(X_train, Y_train)

# Evaluate model (original style)
preds = ensemble.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, preds))
mae = np.mean(np.abs(preds - Y_test))

print(f'\nRMSE: {rmse:.1f} wins')
print(f'MAE: {mae:.1f} wins')

results = df[test_mask][['Season', 'Team']].copy()
results['actual_wins'] = Y_test.values
results['predicted_wins'] = preds.round(1)
results['error'] = (results['predicted_wins'] - results['actual_wins']).round(1)
print("\nRecent Season Results:")
print(results.sort_values(['Season', 'Team']).to_string(index=False))

# Feature Importance
fitted_xgb = ensemble.named_estimators_['xgb']
fitted_rf = ensemble.named_estimators_['rf']

xgb_imp = pd.Series(fitted_xgb.feature_importances_, index=X_cols)
rf_imp = pd.Series(fitted_rf.feature_importances_, index=X_cols)
avg_imp = (xgb_imp + rf_imp) / 2

avg_imp.sort_values().plot(kind='barh', figsize=(10, 8), title='Feature Importance (Averaged)')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nChart saved as feature_importance.png")






