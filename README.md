# Heart Disease Prediction

Binary classification project using the Kaggle Heart Failure Prediction Dataset (918 rows, 12 features).

## Models Used
- Logistic Regression
- Random Forest

## Results
| Model | AUC |
|-------|-----|
| Logistic Regression | 0.93 |
| Random Forest | 0.93 |

## Top Predictive Features
1. ST_Slope
2. Oldpeak
3. Cholesterol
4. MaxHR
5. ChestPainType_ASY

## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## Challenges & Learning

While building this, I focused on understanding how ST_Slope and Cholesterol drive predictions. I used Seaborn to visualize the correlation matrix, which helped identify the strongest predictors.

I used Claude to help with the boilerplate but spent time analyzing the outputs.