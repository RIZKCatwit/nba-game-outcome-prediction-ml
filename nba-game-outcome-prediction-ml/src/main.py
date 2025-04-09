import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

df = pd.read_csv('/content/nba_2024_2025_games_cleaned.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(by='GAME_DATE').reset_index(drop=True)

home_df = df[['GAME_DATE', 'TEAM_NAME_HOME', 'PTS_HOME', 'WL_HOME']].copy()
home_df.rename(columns={
    'TEAM_NAME_HOME': 'TEAM_NAME',
    'PTS_HOME': 'PTS',
    'WL_HOME': 'WL'
}, inplace=True)

away_df = df[['GAME_DATE', 'TEAM_NAME_AWAY', 'PTS_AWAY']].copy()
away_df.rename(columns={
    'TEAM_NAME_AWAY': 'TEAM_NAME',
    'PTS_AWAY': 'PTS'
}, inplace=True)
away_df['WL'] = np.where(df['WL_HOME'] == 'L', 'W', 'L')

team_game_log = pd.concat([home_df, away_df]).sort_values(by='GAME_DATE')
team_game_log['WIN'] = (team_game_log['WL'] == 'W').astype(int)

team_game_log['AVG_PTS_LAST_5'] = team_game_log.groupby('TEAM_NAME')['PTS'].transform(lambda x: x.shift().rolling(5).mean())

def calc_win_streak(x):
    streak = []
    count = 0
    for val in x.shift():
        if val == 1:
            count += 1
        else:
            count = 0
        streak.append(count)
    return pd.Series(streak, index=x.index)

team_game_log['WIN_STREAK_LAST_3'] = team_game_log.groupby('TEAM_NAME')['WIN'].transform(calc_win_streak)

df = df.merge(team_game_log[['TEAM_NAME', 'GAME_DATE', 'AVG_PTS_LAST_5', 'WIN_STREAK_LAST_3']], 
              left_on=['TEAM_NAME_HOME', 'GAME_DATE'], 
              right_on=['TEAM_NAME', 'GAME_DATE'], 
              how='left')
df.rename(columns={
    'AVG_PTS_LAST_5': 'HOME_AVG_PTS_LAST_5',
    'WIN_STREAK_LAST_3': 'HOME_WIN_STREAK'
}, inplace=True)

df = df.merge(team_game_log[['TEAM_NAME', 'GAME_DATE', 'AVG_PTS_LAST_5', 'WIN_STREAK_LAST_3']], 
              left_on=['TEAM_NAME_AWAY', 'GAME_DATE'], 
              right_on=['TEAM_NAME', 'GAME_DATE'], 
              how='left')
df.rename(columns={
    'AVG_PTS_LAST_5': 'AWAY_AVG_PTS_LAST_5',
    'WIN_STREAK_LAST_3': 'AWAY_WIN_STREAK'
}, inplace=True)

df.drop(columns=['TEAM_NAME_x', 'TEAM_NAME_y'], inplace=True)

df['PTS_DIFF_LAST_5'] = df['HOME_AVG_PTS_LAST_5'] - df['AWAY_AVG_PTS_LAST_5']
df['WIN_STREAK_DIFF'] = df['HOME_WIN_STREAK'] - df['AWAY_WIN_STREAK']

features = ['PTS_DIFF_LAST_5', 'WIN_STREAK_DIFF']
df = df.dropna(subset=features + ['HOME_TEAM_WIN'])

X = df[features]
y = df['HOME_TEAM_WIN']

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
probs = pipe.predict_proba(X_test)[:, 1]

print("Logistic Regression with Polynomial Features:")
print(classification_report(y_test, preds))
print(f"AUC Score: {roc_auc_score(y_test, probs):.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Poly)")
plt.show()

RocCurveDisplay.from_predictions(y_test, probs)
plt.title("ROC Curve - Logistic Regression (Poly)")
plt.show()
