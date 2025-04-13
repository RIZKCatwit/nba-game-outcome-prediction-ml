**# nba-game-outcome-prediction-ml
NBA Game Outcome Prediction Using Logistic Regression and Feature Engineering (Spring 2025 | DATA 6250 Final Project)
**NBA Game Outcome Prediction Using Machine Learning
Course: DATA 6250 – Machine Learning for Data Science
Instructor: Dr. Memo Ergezer
Author: Charles Rizk
Date: April 13, 2025

Objective
This project aims to predict whether the home team wins an NBA game using machine learning models trained on 2024–25 regular season data. The workflow includes data preprocessing, feature engineering, baseline modeling, experimentation, and final model selection based on performance and interpretability.

Dataset
Source: Compiled from 2024–25 NBA regular season game logs

Size: ~450+ games

Features: Game-level statistics (points, rebounds, assists, turnovers), along with engineered features

Target: HOME_TEAM_WIN (binary classification)

Strengths
Clean, structured tabular format

Sufficient sample size for classification tasks

Enables derived features like point differentials and win streaks

Weaknesses
No player-level or injury data

Only includes regular-season games (no playoffs or preseason)

Rolling averages are skewed early in the season

Biases
Slight target imbalance (~55% home wins)

No adjustments for opponent strength or team fatigue beyond basic rest-day indicators

Preprocessing and Feature Engineering
Converted and sorted game dates chronologically

Removed null entries and created new features:

REB_DIFF, TOV_DIFF, AST_TOV_HOME, AST_TOV_AWAY

WIN_STREAK_DIFF, PTS_DIFF_LAST_5

Applied StandardScaler and MinMaxScaler for comparative scaling

Baseline Models
Model	Accuracy	AUC Score
Logistic Regression (Poly)	0.65	0.64
Support Vector Machine	0.53	0.43
Random Forest	0.53	0.52
Experiments
Experiment 1: Feature Scaling
StandardScaler and MinMaxScaler were compared.
MinMax scaling produced slightly better results (Accuracy: 0.66, AUC: 0.655).

Experiment 2: Feature Engineering
Basketball-specific domain features were added.
Model performance significantly improved (Accuracy: 0.80, AUC: 0.86).

Experiment 3: PCA (Feature Transformation)
Dimensionality was reduced while retaining 95% of the variance.
Performance remained strong (Accuracy: 0.79, AUC: 0.86).

Experiment 4: Noisy Features
Synthetic continuous and categorical features were added.
Model retained high performance, demonstrating robustness (Accuracy: 0.79, AUC: 0.86).

Experiment 5: Interpretability
SHAP analysis revealed the most influential features:
REB_DIFF, AST_TOV_DIFF, and TOV_DIFF.

Experiment 6: Model Efficiency
Model	Train Time	Predict Time
Logistic Regression	0.0076 sec	0.0016 sec
Random Forest	0.1404 sec	0.0102 sec
Final Model Recommendation
The best-performing model was Logistic Regression with Polynomial Features combined with MinMaxScaler.

Accuracy: 0.80

AUC Score: 0.86

Fast, interpretable, and deployment-ready for real-time applications

Summary of Results
Experiment	Accuracy	AUC Score
Baseline (Polynomial)	0.65	0.64
MinMax Scaled	0.66	0.655
Engineered Features	0.80	0.86
PCA	0.79	0.86
Noise Features Added	0.79	0.86
