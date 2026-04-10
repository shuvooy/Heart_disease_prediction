import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, RocCurveDisplay

sns.set_theme(style='darkgrid', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 130, 'figure.facecolor': 'white'})

FILE_PATH = '/home/shuvooy_/Documents/AI Engineering Project/heart.csv'
label_map = {0: 'No Disease', 1: 'Heart Disease'}
palette   = {0: '#4C72B0', 1: '#DD8452'}

df = pd.read_csv(FILE_PATH)

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.head())

counts = df['HeartDisease'].value_counts().sort_index()
labels = [label_map[i] for i in counts.index]
colors = ['#4C72B0', '#DD8452']

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='black', width=0.45)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 str(v), ha='center', fontweight='bold')
axes[0].set_title('HeartDisease - Count')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, counts.max() * 1.15)

axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('HeartDisease - Proportion')

fig.suptitle('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, hue='HeartDisease',
                  palette=palette, edgecolor='black', ax=axes[i])
    axes[i].set_title(f'{col} vs HeartDisease')
    handles, _ = axes[i].get_legend_handles_labels()
    axes[i].legend(handles, ['No Disease', 'Heart Disease'], title='HeartDisease')

axes[-1].set_visible(False)
fig.suptitle('Categorical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

num_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='HeartDisease',
                 kde=True, palette=palette, alpha=0.55, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

fig.suptitle('Numerical Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

corr = df[num_cols + ['HeartDisease']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, square=True, ax=ax)
ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

numerical_cols   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

num_transformer = Pipeline([('scaler', StandardScaler())])
cat_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore',
                                                       sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

lr_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

y_pred_lr = lr_pipe.predict(X_test)
y_pred_rf = rf_pipe.predict(X_test)

print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Heart Disease']))
print(classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Heart Disease']))

fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(lr_pipe, X_test, y_test,
                               name='Logistic Regression',
                               color='#4C72B0', linewidth=2, ax=ax)
RocCurveDisplay.from_estimator(rf_pipe, X_test, y_test,
                               name='Random Forest',
                               color='#DD8452', linewidth=2, ax=ax)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Chance (AUC = 0.50)')
ax.set_title('ROC Curve - LR vs Random Forest', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.show()

ohe_cols = (rf_pipe.named_steps['preprocessor']
                   .named_transformers_['cat']
                   .named_steps['encoder']
                   .get_feature_names_out(categorical_cols).tolist())

all_cols    = numerical_cols + ohe_cols
importances = rf_pipe.named_steps['classifier'].feature_importances_

feat_df = (pd.DataFrame({'Feature': all_cols, 'Importance': importances})
           .sort_values('Importance', ascending=False)
           .reset_index(drop=True))

fig, ax = plt.subplots(figsize=(9, 7))
sns.barplot(data=feat_df, y='Feature', x='Importance',
            palette=sns.color_palette('Blues_r', n_colors=len(feat_df)),
            edgecolor='black', ax=ax)
ax.set_title('Random Forest - Feature Importances', fontweight='bold')
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.show()

print(feat_df.head())

joblib.dump(rf_pipe, 'heart_model.pkl')

model  = joblib.load('heart_model.pkl')
sample = X_test.iloc[[0]]
pred   = model.predict(sample)[0]
proba  = model.predict_proba(sample)[0]

print(f'actual    : {y_test.iloc[0]} ({label_map[y_test.iloc[0]]})')
print(f'predicted : {pred} ({label_map[pred]})')
print(f'proba     : No Disease = {proba[0]:.3f} | Heart Disease = {proba[1]:.3f}')
