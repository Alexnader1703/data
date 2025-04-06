
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Размер тренировочного набора данных: {train_data.shape}")
print(f"Размер тестового набора данных: {test_data.shape}")

memory_usage_train = train_data.memory_usage(deep=True).sum()
print(f"Размер тренировочного датафрейма в памяти: {memory_usage_train / 1024 ** 2:.2f} МБ")

print("\nТипы данных:")
print(train_data.dtypes)

print("\nОбщая информация о тренировочных данных:")
print(train_data.info())

print("\nПропущенные значения в тренировочных данных:")
print(train_data.isnull().sum())

print("\nПервые 5 строк тренировочных данных:")
print(train_data.head())

numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'id' in numeric_cols:
    numeric_cols.remove('id')
if 'target' in numeric_cols:
    numeric_cols.remove('target')

print("\nСтатистика для числовых переменных:")
stats = train_data[numeric_cols].describe(percentiles=[.25, .5, .75])
print(stats)

categorical_cols = train_data.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nКатегориальные переменные:")
print(categorical_cols)

print("\nРаспределение целевой переменной:")
print(train_data['target'].value_counts())
print(f"Процент положительных примеров: {train_data['target'].mean() * 100:.2f}%")

plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=train_data)
plt.title('Распределение целевой переменной')
plt.savefig('target_distribution.png')
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = train_data[numeric_cols + ['target']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

missing_percentage = train_data.isnull().mean() * 100
print("\nДоля пропущенных значений в каждом столбце (%):")
print(missing_percentage[missing_percentage > 0])

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=train_data[col])
    plt.title(col)
plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()

def handle_outliers(df, cols, lower_quantile=0.01, upper_quantile=0.99):
    df_copy = df.copy()
    for col in cols:
        lower_bound = df_copy[col].quantile(lower_quantile)
        upper_bound = df_copy[col].quantile(upper_quantile)
        df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
    return df_copy


train_data_clean = handle_outliers(train_data, numeric_cols)

print(f"\nКоличество категориальных переменных: {len(categorical_cols)}")

X = train_data_clean.drop(['id', 'target'], axis=1)
y = train_data_clean['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Размер тренировочной выборки: {X_train.shape}")
print(f"Размер валидационной выборки: {X_val.shape}")

numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

knn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

knn_param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

knn_grid_search = GridSearchCV(knn_model, knn_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
knn_grid_search.fit(X_train, y_train)

print(f"\nЛучшие параметры для KNN: {knn_grid_search.best_params_}")
print(f"Лучший результат для KNN (ROC AUC): {knn_grid_search.best_score_:.4f}")

logreg_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

logreg_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__penalty': ['l1', 'l2']
}

logreg_grid_search = GridSearchCV(logreg_model, logreg_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
logreg_grid_search.fit(X_train, y_train)

print(f"\nЛучшие параметры для логистической регрессии: {logreg_grid_search.best_params_}")
print(f"Лучший результат для логистической регрессии (ROC AUC): {logreg_grid_search.best_score_:.4f}")

svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])

svm_param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
svm_grid_search.fit(X_train, y_train)

print(f"\nЛучшие параметры для SVM: {svm_grid_search.best_params_}")
print(f"Лучший результат для SVM (ROC AUC): {svm_grid_search.best_score_:.4f}")


def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"\nРезультаты для {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    fpr, tpr, _ = roc_curve(y, y_prob)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix
    }

knn_best = knn_grid_search.best_estimator_
logreg_best = logreg_grid_search.best_estimator_
svm_best = svm_grid_search.best_estimator_

knn_results = evaluate_model(knn_best, X_val, y_val, "KNN")
logreg_results = evaluate_model(logreg_best, X_val, y_val, "Логистическая регрессия")
svm_results = evaluate_model(svm_best, X_val, y_val, "SVM")

plt.figure(figsize=(10, 8))
plt.plot(knn_results['fpr'], knn_results['tpr'], label=f'KNN (AUC = {knn_results["roc_auc"]:.4f})')
plt.plot(logreg_results['fpr'], logreg_results['tpr'],
         label=f'Логистическая регрессия (AUC = {logreg_results["roc_auc"]:.4f})')
plt.plot(svm_results['fpr'], svm_results['tpr'], label=f'SVM (AUC = {svm_results["roc_auc"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Кривые для разных моделей')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.show()

models = ['KNN', 'Логистическая регрессия', 'SVM']
metrics = {
    'Accuracy': [knn_results['accuracy'], logreg_results['accuracy'], svm_results['accuracy']],
    'Precision': [knn_results['precision'], logreg_results['precision'], svm_results['precision']],
    'Recall': [knn_results['recall'], logreg_results['recall'], svm_results['recall']],
    'F1-score': [knn_results['f1'], logreg_results['f1'], svm_results['f1']],
    'ROC AUC': [knn_results['roc_auc'], logreg_results['roc_auc'], svm_results['roc_auc']]
}

metrics_df = pd.DataFrame(metrics, index=models)
print("\nСравнение метрик для всех моделей:")
print(metrics_df)

best_model_idx = metrics_df['ROC AUC'].idxmax()
best_model_score = metrics_df.loc[best_model_idx, 'ROC AUC']
print(f"\nЛучшая модель по ROC AUC: {best_model_idx} со значением {best_model_score:.4f}")

if best_model_idx == 'KNN':
    best_model = knn_best
elif best_model_idx == 'Логистическая регрессия':
    best_model = logreg_best
else:
    best_model = svm_best

test_predictions = best_model.predict_proba(test_data.drop('id', axis=1))[:, 1]

submission = pd.DataFrame({
    'id': test_data['id'],
    'target': test_predictions
})

submission.to_csv('kidney_stone_predictions.csv', index=False)

print("\nПредсказания для тестового набора данных сохранены в файл 'kidney_stone_predictions.csv'")



"""gravity – удельный вес мочи (плотность мочи)
ph – кислотность (pH) мочи
osmo – осмоляльность мочи (концентрация осмотически активных веществ)
cond – электропроводность мочи
urea – содержание мочевины
calc – содержание кальция"""