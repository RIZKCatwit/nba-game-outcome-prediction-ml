# NBA Game Outcome Prediction Using Machine Learning

**Course**: DATA 6250 – Machine Learning for Data Science  
**Instructor**: Dr. Memo Ergezer  
**Author**: Charles Rizk  
**Date**: April 13, 2025

---

## Objective

This project aims to predict whether the home team wins an NBA game using machine learning models trained on 2024–25 regular season data. The workflow includes data preprocessing, feature engineering, baseline modeling, experimentation, and final model selection based on performance and interpretability.

---

## Dataset

- **Source**: Compiled from 2024–25 NBA regular season game logs  
- **Size**: Over 450 games  
- **Features**: Game-level statistics (points, rebounds, assists, turnovers), plus engineered features  
- **Target Variable**: `HOME_TEAM_WIN` (binary classification: 1 = win, 0 = loss)

### Strengths
- Clean, structured tabular format
- Sufficient sample size for classification tasks
- Enables derived features such as win streaks and scoring differentials

### Weaknesses
- No player-level data (e.g., injuries, rotations, matchups)
- Does not include playoff or preseason games
- Rolling averages may be skewed early in the season

### Biases
- Slight target imbalance (~55% home wins)
- Lacks adjustments for opponent strength or player fatigue beyond basic rest metrics

---

## Preprocessing and Feature Engineering

- Converted and sorted game dates
- Removed null entries and engineered new features including:
  - `REB_DIFF`, `TOV_DIFF`, `AST_TOV_HOME`, `AST_TOV_AWAY`
  - `WIN_STREAK_DIFF`, `PTS_DIFF_LAST_5`
- Applied both `StandardScaler` and `MinMaxScaler` for model comparisons

---

## Baseline Models

| Model                      | Accuracy | AUC Score |
|----------------------------|----------|-----------|
| Logistic Regression (Poly) | 0.65     | 0.64      |
| Support Vector Machine     | 0.53     | 0.43      |
| Random Forest              | 0.53     | 0.52      |

---

## Experiments

### Experiment 1: Feature Scaling  
Compared `StandardScaler` vs `MinMaxScaler`.  
MinMaxScaler led to slightly better performance (Accuracy: 0.66, AUC: 0.655).

### Experiment 2: Feature Engineering  
Added basketball-specific features like `REB_DIFF` and assist-to-turnover ratios.  
Model performance improved significantly (Accuracy: 0.80, AUC: 0.86).

### Experiment 3: PCA (Feature Transformation)  
Reduced dimensionality while retaining 95% of variance.  
Performance remained strong (Accuracy: 0.79, AUC: 0.86).

### Experiment 4: Noisy Features  
Introduced random continuous and categorical features.  
Model performance stayed consistent, indicating good generalization (Accuracy: 0.79, AUC: 0.86).

### Experiment 5: Model Interpretability  
Used SHAP to identify important features:  
Top contributors included `REB_DIFF`, `AST_TOV_DIFF`, and `TOV_DIFF`.

### Experiment 6: Efficiency  
Measured training and inference time for baseline models.

| Model                | Train Time | Predict Time |
|----------------------|------------|--------------|
| Logistic Regression  | 0.0076 sec | 0.0016 sec   |
| Random Forest        | 0.1404 sec | 0.0102 sec   |

---

## Final Model Recommendation

The best-performing model was **Logistic Regression with Polynomial Features** and **MinMaxScaler**.

- Accuracy: 0.80  
- AUC Score: 0.86  
- Fast, lightweight, interpretable, and suitable for deployment

---

## Summary of Results

| Experiment              | Accuracy | AUC Score |
|-------------------------|----------|-----------|
| Baseline (Polynomial)   | 0.65     | 0.64      |
| MinMax Scaled           | 0.66     | 0.655     |
| Engineered Features     | 0.80     | 0.86      |
| PCA                     | 0.79     | 0.86      |
| Noisy Features Added    | 0.79     | 0.86      |

---

## Future Improvements

- Experiment with ensemble models (e.g., XGBoost, LightGBM)
- Integrate player-level data (injuries, minutes, matchups)
- Build a deployable web application for live prediction

---

## Repository Structure

```
nba-outcome-prediction/
├── images/
│   ├── confusion_matrix_poly.png
│   ├── roc_curve_poly.png
│   └── ...
├── nba_2024_2025_games_cleaned.csv
├── final_model_results.ipynb
├── README.md
```

---

## Acknowledgments

Special thanks to Dr. Memo Ergezer for his continued support and insightful feedback throughout the semester. This project was inspired by the intersection of data science and real-world sports analytics.
