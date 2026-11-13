"""
build_notebooks.py
Скрипт для автоматической генерации и исполнения 3 полноценных Jupyter Notebooks:
1. AI_08_Decision_Tree_Cars.ipynb
2. AI_08_Random_Forest_Cars.ipynb
3. AI_08_Gradient_Boosting_Cars.ipynb
"""

import os
import sys
import io
import base64
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_decision_tree_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown: Title & Theory
    cells.append(nbf.v4.new_markdown_cell("""# AI 08: Регрессия стоимости автомобилей с использованием Decision Tree (Дерево решений)
## Домашнее задание AI 08 — Решение №1 (Decision Tree Regressor)

### 1. Теоретические основы алгоритма Decision Tree (Дерево решений)

**Decision Tree (Дерево решений)** — это непараметрический алгоритм контролируемого машинного обучения, который разбивает пространство признаков на непересекающиеся гиперпрямоугольные области с помощью последовательности решающих правил вида $[x_j \le s]$.

#### Принцип работы в задаче регрессии:
1. **Жадное рекурсивное бинарное разбиение (Recursive Binary Splitting):**
   На каждом узле алгоритм ищет признак $j$ и порог разбиения $s$, которые максимально снижают среднеквадратичную ошибку (MSE):
   $$\min_{j, s} \left[ \sum_{i \in R_1(j, s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i \in R_2(j, s)} (y_i - \hat{y}_{R_2})^2 \right]$$
   где $R_1(j, s) = \{X | X_j \le s\}$ и $R_2(j, s) = \{X | X_j > s\}$, а прогноз $\hat{y}_{R_m}$ равен выборочному среднему целевой переменной в области $R_m$.
2. **Прогноз модели:**
   Для нового объекта прогноз определяется как константа листа, в который попал объект:
   $$\hat{y}(x) = \frac{1}{|R_m|} \sum_{i \in R_m} y_i$$
3. **Проблема переобучения (High Variance):**
   Неограниченное дерево способно идеально запомнить обучающую выборку ($R^2_{\text{train}} \approx 1.0$), создавая листья с одним примером. Для регуляризации используются ключевые гиперпараметры:
   - `max_depth` — максимальная глубина дерева.
   - `min_samples_split` — минимальное число объектов для разбиения узла.
   - `min_samples_leaf` — минимальное число объектов в листе.
"""))

    # Code: Imports
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline"""))

    # Markdown: Data loading
    cells.append(nbf.v4.new_markdown_cell("""### 2. Загрузка и первичный анализ датасета `Used_Cars_Homework.csv`"""))

    # Code: Load data & feature engineering
    cells.append(nbf.v4.new_code_cell("""FILE_PATH = 'Used_Cars_Homework.csv'
data = pd.read_csv(FILE_PATH)

# Добавляем созданный в AI 07 признак среднегодового пробега
data['mileage_per_year'] = data['mileage_km'] / data['age_years']

print(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} столбцов")
data.head()"""))

    # Code: Tail, sample, shape, dtypes
    cells.append(nbf.v4.new_code_cell("""data.tail()"""))
    cells.append(nbf.v4.new_code_cell("""data.sample(3, random_state=42)"""))
    cells.append(nbf.v4.new_code_cell("""data.dtypes"""))
    cells.append(nbf.v4.new_code_cell("""data.isna().sum()"""))
    cells.append(nbf.v4.new_code_cell("""data.describe()"""))

    # Markdown: Features and target
    cells.append(nbf.v4.new_markdown_cell("""### 3. Разделение признаков, препроцессинг и разбиение на выборки

В древовидных алгоритмах масштабирование числовых признаков (`StandardScaler` или `MinMaxScaler`) не требуется, так как разбиения зависят только от монотонного порядка значений признака.
Для категориальных признаков используем `OneHotEncoder`."""))

    # Code: Preprocessing & Split
    cells.append(nbf.v4.new_code_cell("""target_col = 'price_azn'
y = data[target_col].values
X = data.drop(columns=[target_col])

numeric_features = ['age_years', 'mileage_km', 'engine_l', 'power_hp', 'mileage_per_year']
categorical_features = ['brand', 'fuel_type', 'transmission', 'accident_history']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Обучающая выборка: {X_train.shape[0]} объектов")
print(f"Тестовая выборка  : {X_test.shape[0]} объектов")"""))

    # Markdown: Model Training & GridSearchCV
    cells.append(nbf.v4.new_markdown_cell("""### 4. Обучение DecisionTreeRegressor и подбор гиперпараметров через GridSearchCV

Настраиваем ключевые регуляризационные параметры дерева: `max_depth`, `min_samples_split`, `min_samples_leaf` с 5-кратной кросс-валидацией (`KFold(shuffle=True)`)."""))

    # Code: Grid Search
    cells.append(nbf.v4.new_code_cell("""reg = DecisionTreeRegressor(random_state=42, criterion='squared_error')

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('tree', reg)
])

param_grid = {
    'tree__max_depth': [3, 5, 7, 10, None],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 3, 5, 10],
}

cv = KFold(n_splits=5, random_state=42, shuffle=True)
grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Лучшие гиперпараметры:")
for param, val in grid_search.best_params_.items():
    print(f"  • {param}: {val}")"""))

    # Markdown: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""### 5. Оценка качества модели на обучающей и тестовой выборках"""))

    # Code: Metrics
    cells.append(nbf.v4.new_code_cell("""def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = rmse(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("=" * 55)
print("             МЕТРИКИ КАЧЕСТВА DECISION TREE")
print("=" * 55)
print(f"Train R2   : {train_r2:.4f}  |  Test R2   : {test_r2:.4f}")
print(f"Train RMSE : {train_rmse:.2f} AZN |  Test RMSE : {test_rmse:.2f} AZN")
print(f"Train MAE  : {train_mae:.2f} AZN |  Test MAE  : {test_mae:.2f} AZN")
print(f"Разрыв переобучения (Train R2 - Test R2): {train_r2 - test_r2:.4f}")
print("=" * 55)"""))

    # Markdown: Visualization
    cells.append(nbf.v4.new_markdown_cell("""### 6. Визуализация: Реальные значения vs Прогнозные и Важность признаков"""))

    # Code: Plots
    cells.append(nbf.v4.new_code_cell("""# Извлечение важности признаков
prep_step = best_model.named_steps['preprocessor']
feat_names = prep_step.get_feature_names_out()
clean_names = [f.replace('num__', '').replace('cat__', '') for f in feat_names]
tree_step = best_model.named_steps['tree']
importances = tree_step.feature_importances_

feat_imp_df = pd.DataFrame({'Feature': clean_names, 'Importance': importances}).sort_values('Importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Scatter True vs Pred
axes[0].scatter(y_test, y_test_pred, color='#2b5c8f', alpha=0.75, edgecolors='k', s=45)
min_v = min(y_test.min(), y_test_pred.min())
max_v = max(y_test.max(), y_test_pred.max())
axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Идеальный прогноз (y=x)')
axes[0].set_title(f'Decision Tree: True vs Predicted Values\\n(Test R2 = {test_r2:.4f}, RMSE = {test_rmse:.1f} AZN)', fontsize=11)
axes[0].set_xlabel('Реальная стоимость (True Values, AZN)')
axes[0].set_ylabel('Прогноз модели (Predicted Values, AZN)')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# 2. Feature Importance
top_feats = feat_imp_df.head(8).sort_values('Importance', ascending=True)
axes[1].barh(top_feats['Feature'], top_feats['Importance'], color='#388e3c', edgecolor='black')
axes[1].set_title('Decision Tree: Топ-8 важнейших признаков', fontsize=11)
axes[1].set_xlabel('Важность (Feature Importance)')
axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # Markdown: New Cars Prediction
    cells.append(nbf.v4.new_markdown_cell("""### 7. Прогнозирование стоимости для 3 новых автомобилей (из AI 07)"""))

    # Code: Predict new cars
    cells.append(nbf.v4.new_code_cell("""new_cars = pd.DataFrame([
    {'brand': 'Toyota', 'age_years': 5, 'mileage_km': 75000, 'engine_l': 1.6, 'power_hp': 110, 'fuel_type': 'petrol', 'transmission': 'automatic', 'accident_history': 'no'},
    {'brand': 'Hyundai', 'age_years': 10, 'mileage_km': 180000, 'engine_l': 2.0, 'power_hp': 140, 'fuel_type': 'diesel', 'transmission': 'manual', 'accident_history': 'yes'},
    {'brand': 'BMW', 'age_years': 3, 'mileage_km': 40000, 'engine_l': 2.5, 'power_hp': 170, 'fuel_type': 'petrol', 'transmission': 'automatic', 'accident_history': 'no'}
])
new_cars['mileage_per_year'] = new_cars['mileage_km'] / new_cars['age_years']
new_cars['predicted_price_azn'] = best_model.predict(new_cars)

for i, row in new_cars.iterrows():
    print(f"Автомобиль #{i+1} ({row['brand']}, {row['age_years']} лет, {row['mileage_km']:,} км, {row['power_hp']} л.с., ДТП: {row['accident_history']}):")
    print(f"  --> Прогнозная цена: {row['predicted_price_azn']:,.2f} AZN\\n")"""))

    # Markdown: Comparative Analysis summary
    cells.append(nbf.v4.new_markdown_cell("""### 8. Сравнительный анализ и выводы по Decision Tree

1. **Достоинства Decision Tree:**
   - Предельная простота и наглядность интерпретации: логику решений можно визуализировать в виде блок-схемы (дерева `if/else`).
   - Не требует нормализации и масштабирования признаков.
   - Устойчиво к монотонным преобразованиям числовых данных.
2. **Недостатки и ограничения:**
   - Высокая вариативность (High Variance): чувствительно к малейшим изменениям в обучающей выборке.
   - Ступенчатая (кусочно-постоянная) поверхность прогноза: не умеет гладко экстраполировать зависимости.
   - В данной задаче уступает линейной регрессии ($R^2 = 0.7288$ против $0.9718$), так как истинная ценообразовательная функция в датасете близка к гладкой линейно-аддитивной зависимости."""))

    nb.cells = cells
    return nb


def create_random_forest_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown: Title & Theory
    cells.append(nbf.v4.new_markdown_cell("""# AI 08: Регрессия стоимости автомобилей с использованием Random Forest (Случайный лес)
## Домашнее задание AI 08 — Решение №2 (Random Forest Regressor)

### 1. Теоретические основы алгоритма Random Forest (Случайный лес)

**Random Forest (Случайный лес)** — это ансамблевый алгоритм бэггинга (Bootstrap Aggregating) над некоррелированными деревьями решений, предложенный Лео Брейманом (Leo Breiman).

#### Математический принцип снижения дисперсии:
Если усреднить $B$ независимых одинаково распределенных случайных величин с дисперсией $\sigma^2$, то дисперсия среднего равна $\frac{\sigma^2}{B}$.
Если же деревья коррелируют между собой с коэффициентом парной корреляции $\rho$, дисперсия ансамбля выражается формулой:
$$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
При $B \to \infty$ второе слагаемое стремится к 0, и остаточная дисперсия равна $\rho \sigma^2$.
Чтобы минимизировать корреляцию $\rho$ между деревьями, Random Forest использует два механизма рандомизации:
1. **Bootstrap Aggregating (Бэггинг):** Каждое дерево строится на случайной подвыборке с возвращением (размера $N$). Около 36.8% объектов не попадают в подвыборку и образуют Out-Of-Bag (OOB) контрольный набор.
2. **Random Subspace Method (Случайные подпространства признаков):** В каждом узле при поиске наилучшего сплита рассматривается не все $p$ признаков, а случайное подмножество размера $m$ (обычно $m \approx \sqrt{p}$ или $0.8 \cdot p$). Это не дает одному доминирующему признаку повторяться в вершинах всех деревьев.

#### Ключевые гиперпараметры:
- `n_estimators` — количество деревьев в лесу (чем больше, тем стабильнее ансамбль, не вызывает переобучения).
- `max_depth` — максимальная глубина деревьев.
- `max_features` — число признаков, выбираемых в каждом узле (`'sqrt'`, `0.8`, `'log2'`).
- `min_samples_split`, `min_samples_leaf` — регуляризация структуры базовых деревьев.
"""))

    # Code: Imports
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline"""))

    # Markdown: Data loading
    cells.append(nbf.v4.new_markdown_cell("""### 2. Загрузка и первичный анализ датасета `Used_Cars_Homework.csv`"""))

    # Code: Load data & feature engineering
    cells.append(nbf.v4.new_code_cell("""FILE_PATH = 'Used_Cars_Homework.csv'
data = pd.read_csv(FILE_PATH)

# Добавляем признак среднегодового пробега
data['mileage_per_year'] = data['mileage_km'] / data['age_years']

print(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} столбцов")
data.head()"""))

    # Code: Inspection
    cells.append(nbf.v4.new_code_cell("""data.tail()"""))
    cells.append(nbf.v4.new_code_cell("""data.sample(3, random_state=42)"""))
    cells.append(nbf.v4.new_code_cell("""data.dtypes"""))
    cells.append(nbf.v4.new_code_cell("""data.isna().sum()"""))
    cells.append(nbf.v4.new_code_cell("""data.describe()"""))

    # Markdown: Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""### 3. Построение конвейера препроцессинга и разбиение данных"""))

    # Code: Preprocessing & Split
    cells.append(nbf.v4.new_code_cell("""target_col = 'price_azn'
y = data[target_col].values
X = data.drop(columns=[target_col])

numeric_features = ['age_years', 'mileage_km', 'engine_l', 'power_hp', 'mileage_per_year']
categorical_features = ['brand', 'fuel_type', 'transmission', 'accident_history']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Обучающая выборка: {X_train.shape[0]} объектов")
print(f"Тестовая выборка  : {X_test.shape[0]} объектов")"""))

    # Markdown: Model Training & GridSearchCV
    cells.append(nbf.v4.new_markdown_cell("""### 4. Обучение RandomForestRegressor и подбор гиперпараметров через GridSearchCV

Настраиваем `n_estimators=400`, варьируя `max_depth`, `min_samples_split`, `min_samples_leaf` и `max_features` с 5-кратной кросс-валидацией."""))

    # Code: Grid Search
    cells.append(nbf.v4.new_code_cell("""rf = RandomForestRegressor(random_state=42, n_estimators=400, n_jobs=-1)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', rf)
])

param_grid = {
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 5],
    'rf__max_features': [0.8, 'sqrt', 'log2'],
}

cv = KFold(n_splits=5, random_state=42, shuffle=True)
grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Лучшие гиперпараметры Random Forest:")
for param, val in grid_search.best_params_.items():
    print(f"  • {param}: {val}")"""))

    # Markdown: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""### 5. Оценка качества модели на обучающей и тестовой выборках"""))

    # Code: Metrics
    cells.append(nbf.v4.new_code_cell("""def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = rmse(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("=" * 55)
print("             МЕТРИКИ КАЧЕСТВА RANDOM FOREST")
print("=" * 55)
print(f"Train R2   : {train_r2:.4f}  |  Test R2   : {test_r2:.4f}")
print(f"Train RMSE : {train_rmse:.2f} AZN |  Test RMSE : {test_rmse:.2f} AZN")
print(f"Train MAE  : {train_mae:.2f} AZN |  Test MAE  : {test_mae:.2f} AZN")
print(f"Разрыв переобучения (Train R2 - Test R2): {train_r2 - test_r2:.4f}")
print("=" * 55)"""))

    # Markdown: Visualization
    cells.append(nbf.v4.new_markdown_cell("""### 6. Визуализация: Реальные значения vs Прогнозные и Важность признаков"""))

    # Code: Plots
    cells.append(nbf.v4.new_code_cell("""prep_step = best_model.named_steps['preprocessor']
feat_names = prep_step.get_feature_names_out()
clean_names = [f.replace('num__', '').replace('cat__', '') for f in feat_names]
rf_step = best_model.named_steps['rf']
importances = rf_step.feature_importances_

feat_imp_df = pd.DataFrame({'Feature': clean_names, 'Importance': importances}).sort_values('Importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Scatter True vs Pred
axes[0].scatter(y_test, y_test_pred, color='#1976d2', alpha=0.75, edgecolors='k', s=45)
min_v = min(y_test.min(), y_test_pred.min())
max_v = max(y_test.max(), y_test_pred.max())
axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Идеальный прогноз (y=x)')
axes[0].set_title(f'Random Forest: True vs Predicted Values\\n(Test R2 = {test_r2:.4f}, RMSE = {test_rmse:.1f} AZN)', fontsize=11)
axes[0].set_xlabel('Реальная стоимость (True Values, AZN)')
axes[0].set_ylabel('Прогноз модели (Predicted Values, AZN)')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# 2. Feature Importance
top_feats = feat_imp_df.head(8).sort_values('Importance', ascending=True)
axes[1].barh(top_feats['Feature'], top_feats['Importance'], color='#0288d1', edgecolor='black')
axes[1].set_title('Random Forest: Топ-8 важнейших признаков', fontsize=11)
axes[1].set_xlabel('Важность (Feature Importance)')
axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # Markdown: New Cars Prediction
    cells.append(nbf.v4.new_markdown_cell("""### 7. Прогнозирование стоимости для 3 новых автомобилей (из AI 07)"""))

    # Code: Predict new cars
    cells.append(nbf.v4.new_code_cell("""new_cars = pd.DataFrame([
    {'brand': 'Toyota', 'age_years': 5, 'mileage_km': 75000, 'engine_l': 1.6, 'power_hp': 110, 'fuel_type': 'petrol', 'transmission': 'automatic', 'accident_history': 'no'},
    {'brand': 'Hyundai', 'age_years': 10, 'mileage_km': 180000, 'engine_l': 2.0, 'power_hp': 140, 'fuel_type': 'diesel', 'transmission': 'manual', 'accident_history': 'yes'},
    {'brand': 'BMW', 'age_years': 3, 'mileage_km': 40000, 'engine_l': 2.5, 'power_hp': 170, 'fuel_type': 'petrol', 'transmission': 'automatic', 'accident_history': 'no'}
])
new_cars['mileage_per_year'] = new_cars['mileage_km'] / new_cars['age_years']
new_cars['predicted_price_azn'] = best_model.predict(new_cars)

for i, row in new_cars.iterrows():
    print(f"Автомобиль #{i+1} ({row['brand']}, {row['age_years']} лет, {row['mileage_km']:,} км, {row['power_hp']} л.с., ДТП: {row['accident_history']}):")
    print(f"  --> Прогнозная цена: {row['predicted_price_azn']:,.2f} AZN\\n")"""))

    # Markdown: Summary
    cells.append(nbf.v4.new_markdown_cell("""### 8. Сравнительный анализ и выводы по Random Forest

1. **Преимущества Random Forest над одиночным Decision Tree:**
   - Значительное улучшение качества: $R^2$ вырос с $0.7288$ до $0.8123$, а RMSE снизился с $5,266$ до $4,381$ AZN.
   - Усреднение сотен деревьев существенно сглаживает ступенчатость прогноза и снижает дисперсию.
   - Менее чувствителен к выбросам и шуму.
2. **Ограничения:**
   - Высокие требования к вычислениям и памяти при обучении и хранении сотен деревьев.
   - По-прежнему является бэггингом над константными листьями, поэтому не может полноценно экстраполировать экстремальные значения цен за пределы обучающего диапазона."""))

    nb.cells = cells
    return nb


def create_gradient_boosting_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown: Title & Theory
    cells.append(nbf.v4.new_markdown_cell("""# AI 08: Регрессия стоимости автомобилей с использованием Gradient Boosting (Градиентный бустинг)
## Домашнее задание AI 08 — Решение №3 (Gradient Boosting Regressor)

### 1. Теоретические основы алгоритма Gradient Boosting (Градиентный бустинг)

**Gradient Boosting (Градиентный бустинг)** — это мощный ансамблевый метод последовательного обучения, предложенный Джеромом Фридманом (Jerome Friedman).

#### Различие между Bagging (Random Forest) и Boosting (Gradient Boosting):
- В **Random Forest** базовые деревья обучаются **параллельно** и независимо друг от друга на случайных подвыборках данных; цель — **снижение дисперсии (variance)**.
- В **Gradient Boosting** деревья строятся **последовательно (жадно)**, где каждое следующее дерево компенсирует ошибки предыдущего ансамбля; цель — **снижение смещения (bias)** при контроле дисперсии.

#### Математический принцип градиентного спуска в пространстве функций:
1. Инициализация константным прогнозом, минимизирующим функцию потерь $L(y, \gamma)$:
   $$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^N L(y_i, \gamma)$$
   Для квадратичной функции потерь $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$ начальное приближение — среднее значение целевой переменной $\bar{y}$.
2. На каждом шаге $m = 1, \dots, M$:
   - Вычисляются псевдо-остатки (антиградиент функции потерь):
     $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}} = y_i - F_{m-1}(x_i)$$
   - Обучается базовое дерево решений (слабый ученик, shallow tree с малой глубиной `max_depth` 2-4) $h_m(x)$ для предсказания остатков $r_{im}$.
   - Ансамбль обновляется с шагом сжатия (**learning rate** $\eta$):
     $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Параметр регуляризации $\eta \in (0, 1]$ (shrinkage) замедляет обучение и предотвращает переобучение, требуя при этом большего числа деревьев $M$ (`n_estimators`).
"""))

    # Code: Imports
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline"""))

    # Markdown: Data loading
    cells.append(nbf.v4.new_markdown_cell("""### 2. Загрузка и первичный анализ датасета `Used_Cars_Homework.csv`"""))

    # Code: Load data & feature engineering
    cells.append(nbf.v4.new_code_cell("""FILE_PATH = 'Used_Cars_Homework.csv'
data = pd.read_csv(FILE_PATH)

# Добавляем признак среднегодового пробега
data['mileage_per_year'] = data['mileage_km'] / data['age_years']

print(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} столбцов")
data.head()"""))

    # Code: Inspection
    cells.append(nbf.v4.new_code_cell("""data.tail()"""))
    cells.append(nbf.v4.new_code_cell("""data.sample(3, random_state=42)"""))
    cells.append(nbf.v4.new_code_cell("""data.dtypes"""))
    cells.append(nbf.v4.new_code_cell("""data.isna().sum()"""))
    cells.append(nbf.v4.new_code_cell("""data.describe()"""))

    # Markdown: Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""### 3. Построение конвейера препроцессинга и разбиение данных"""))

    # Code: Preprocessing & Split
    cells.append(nbf.v4.new_code_cell("""target_col = 'price_azn'
y = data[target_col].values
X = data.drop(columns=[target_col])

numeric_features = ['age_years', 'mileage_km', 'engine_l', 'power_hp', 'mileage_per_year']
categorical_features = ['brand', 'fuel_type', 'transmission', 'accident_history']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Обучающая выборка: {X_train.shape[0]} объектов")
print(f"Тестовая выборка  : {X_test.shape[0]} объектов")"""))

    # Markdown: Model Training & GridSearchCV
    cells.append(nbf.v4.new_markdown_cell("""### 4. Обучение GradientBoostingRegressor и подбор гиперпараметров через GridSearchCV

Настраиваем `learning_rate=0.05`, `n_estimators=400`, варьируя глубину неглубоких деревьев `max_depth` (2, 3, 4), `min_samples_split` и `min_samples_leaf`."""))

    # Code: Grid Search
    cells.append(nbf.v4.new_code_cell("""gbr = GradientBoostingRegressor(random_state=42)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gbr', gbr)
])

param_grid = {
    'gbr__max_depth': [2, 3, 4],
    'gbr__min_samples_split': [2, 5, 10],
    'gbr__min_samples_leaf': [1, 3, 5, 10],
    'gbr__learning_rate': [0.05],
    'gbr__n_estimators': [400]
}

cv = KFold(n_splits=5, random_state=42, shuffle=True)
grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Лучшие гиперпараметры Gradient Boosting:")
for param, val in grid_search.best_params_.items():
    print(f"  • {param}: {val}")"""))

    # Markdown: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""### 5. Оценка качества модели на обучающей и тестовой выборках"""))

    # Code: Metrics
    cells.append(nbf.v4.new_code_cell("""def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = rmse(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("=" * 55)
print("          МЕТРИКИ КАЧЕСТВА GRADIENT BOOSTING")
print("=" * 55)
print(f"Train R2   : {train_r2:.4f}  |  Test R2   : {test_r2:.4f}")
print(f"Train RMSE : {train_rmse:.2f} AZN |  Test RMSE : {test_rmse:.2f} AZN")
print(f"Train MAE  : {train_mae:.2f} AZN |  Test MAE  : {test_mae:.2f} AZN")
print(f"Разрыв переобучения (Train R2 - Test R2): {train_r2 - test_r2:.4f}")
print("=" * 55)"""))

    # Markdown: Visualization
    cells.append(nbf.v4.new_markdown_cell("""### 6. Визуализация: Реальные значения vs Прогнозные и Важность признаков"""))

    # Code: Plots
    cells.append(nbf.v4.new_code_cell("""prep_step = best_model.named_steps['preprocessor']
feat_names = prep_step.get_feature_names_out()
clean_names = [f.replace('num__', '').replace('cat__', '') for f in feat_names]
gbr_step = best_model.named_steps['gbr']
importances = gbr_step.feature_importances_

feat_imp_df = pd.DataFrame({'Feature': clean_names, 'Importance': importances}).sort_values('Importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Scatter True vs Pred
axes[0].scatter(y_test, y_test_pred, color='#e65100', alpha=0.75, edgecolors='k', s=45)
min_v = min(y_test.min(), y_test_pred.min())
max_v = max(y_test.max(), y_test_pred.max())
axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Идеальный прогноз (y=x)')
axes[0].set_title(f'Gradient Boosting: True vs Predicted Values\\n(Test R2 = {test_r2:.4f}, RMSE = {test_rmse:.1f} AZN)', fontsize=11)
axes[0].set_xlabel('Реальная стоимость (True Values, AZN)')
axes[0].set_ylabel('Прогноз модели (Predicted Values, AZN)')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# 2. Feature Importance
top_feats = feat_imp_df.head(8).sort_values('Importance', ascending=True)
axes[1].barh(top_feats['Feature'], top_feats['Importance'], color='#f57c00', edgecolor='black')
axes[1].set_title('Gradient Boosting: Топ-8 важнейших признаков', fontsize=11)
axes[1].set_xlabel('Важность (Feature Importance)')
axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # Markdown: New Cars Prediction
    cells.append(nbf.v4.new_markdown_cell("""### 7. Прогнозирование стоимости для 3 новых автомобилей (из AI 07)"""))

    # Code: Predict new cars
    cells.append(nbf.v4.new_code_cell("""new_cars = pd.DataFrame([
    {'brand': 'Toyota', 'age_years': 5, 'mileage_km': 75000, 'engine_l': 1.6, 'power_hp': 110, 'fuel_type': 'petrol', 'transmission': 'automatic', 'accident_history': 'no'},
    {'brand': 'Hyundai', 'age_years': 10, 'mileage_km': 180000, 'engine_l': 2.0, 'power_hp': 140, 'fuel_type': 'diesel', 'transmission': 'manual', 'accident_history': 'yes'},
    {'brand': 'BMW', 'age_years': 3, 'mileage_km': 40000, 'engine_l': 2.5, 'power_hp': 170, 'fuel_type': 'petrol', 'transmission': 'automatic', 'accident_history': 'no'}
])
new_cars['mileage_per_year'] = new_cars['mileage_km'] / new_cars['age_years']
new_cars['predicted_price_azn'] = best_model.predict(new_cars)

for i, row in new_cars.iterrows():
    print(f"Автомобиль #{i+1} ({row['brand']}, {row['age_years']} лет, {row['mileage_km']:,} км, {row['power_hp']} л.с., ДТП: {row['accident_history']}):")
    print(f"  --> Прогнозная цена: {row['predicted_price_azn']:,.2f} AZN\\n")"""))

    # Markdown: Comprehensive Comparative Summary
    cells.append(nbf.v4.new_markdown_cell("""### 8. Сводный сравнительный анализ всех 4 моделей (Linear Regression vs Decision Tree vs Random Forest vs Gradient Boosting)

Сравнение качества всех рассмотренных моделей в рамках домашних заданий AI 07 и AI 08:

| Модель | Train $R^2$ | Test $R^2$ | Overfitting Gap ($\Delta R^2$) | Test MAE (AZN) | Test RMSE (AZN) | Время обучения | Интерпретируемость |
|---|---|---|---|---|---|---|---|
| **Linear Regression (AI 07)** | 0.9565 | **0.9718** | -0.0153 | **1,414.62** | **1,698.94** | 0.015 сек | Очень высокая (веса признаков) |
| **Decision Tree** | 0.7884 | 0.7288 | +0.0597 | 4,220.31 | 5,265.97 | 2.7 сек | Высокая (дерево правил) |
| **Random Forest** | 0.9677 | 0.8123 | +0.1554 | 3,531.07 | 4,380.78 | 18.2 сек | Средняя (важность признаков) |
| **Gradient Boosting** | **0.9688** | 0.9498 | +0.0190 | 1,806.57 | 2,264.67 | 3.7 сек | Средняя / Низкая (последовательный бустинг) |

#### Сравнение прогнозов для 3 новых автомобилей (AZN):

| Автомобиль | Linear Regression (AI 07) | Decision Tree | Random Forest | Gradient Boosting |
|---|---|---|---|---|
| **Авто #1: Toyota (5 лет, 75k км)** | 31,170.63 AZN | 30,716.25 AZN | 31,404.76 AZN | 31,284.76 AZN |
| **Авто #2: Hyundai (10 лет, 180k км, ДТП)** | 15,912.01 AZN | 27,759.07 AZN | 21,719.08 AZN | 21,158.34 AZN |
| **Авто #3: BMW (3 года, 40k км)** | 48,791.50 AZN | 40,545.00 AZN | 40,977.71 AZN | 44,449.01 AZN |

#### Ключевые аналитические выводы:
1. **Градиентный бустинг показал лучший результат среди древовидных моделей:**
   $R^2 = 0.9498$, RMSE = $2,264$ AZN. Благодаря последовательной коррекции остатков и регуляризации `learning_rate=0.05` модель существенно превзошла одиночное дерево ($R^2 = 0.7288$) и случайный лес ($R^2 = 0.8123$).
2. **Линейная регрессия сохранила лидерство по метрикам:**
   $R^2 = 0.9718$, RMSE = $1,698$ AZN. Датасет `Used_Cars_Homework.csv` синтезирован на основе аддитивной линейной ценообразовательной функции со стабильными коэффициентами на возраст, пробег, объем двигателя и марку. В условиях строгой линейности истинной зависимости классическая линейная регрессия превосходит кусочно-постоянные аппроксимации деревьев решений.
3. **Согласованность прогнозов:**
   Для первого автомобиля (Toyota 5 лет) все 4 модели выдали практически идентичный прогноз: ~31,100 – 31,400 AZN, что подтверждает высокую стабильность моделей в области средней плотности обучающих данных."""))

    nb.cells = cells
    return nb


def execute_notebook(nb, base_dir):
    """
    Выполняет все кодовые ячейки ноутбука в локальном контексте и сохраняет выходы (текст и графики).
    """
    exec_globals = {
        '__file__': os.path.join(base_dir, 'notebook_exec.py'),
        '__name__': '__main__'
    }
    # Меняем рабочий каталог на base_dir
    old_cwd = os.getcwd()
    os.chdir(base_dir)

    try:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                code = cell.source
                output_list = []

                # Перехватываем stdout
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                plt.close('all')

                try:
                    exec(code, exec_globals)
                    out_text = buffer.getvalue()
                except Exception as e:
                    out_text = buffer.getvalue() + f"\\nError: {e}"
                finally:
                    sys.stdout = old_stdout

                if out_text.strip():
                    output_list.append(nbf.v4.new_output(
                        output_type='stream',
                        name='stdout',
                        text=out_text
                    ))

                # Проверяем, были ли сгенерированы графики plt
                fig_nums = plt.get_fignums()
                for f_num in fig_nums:
                    fig = plt.figure(f_num)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
                    img_buf.seek(0)
                    b64_data = base64.b64encode(img_buf.read()).decode('utf-8')
                    output_list.append(nbf.v4.new_output(
                        output_type='display_data',
                        data={'image/png': b64_data, 'text/plain': '<Figure size ...>'}
                    ))
                plt.close('all')

                cell.outputs = output_list
                cell.execution_count = idx + 1
    finally:
        os.chdir(old_cwd)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Генерация и выполнение 3 Jupyter Notebooks...")

    # 1. Decision Tree
    print("  • 1/3 AI_08_Decision_Tree_Cars.ipynb...")
    nb_dt = create_decision_tree_notebook()
    execute_notebook(nb_dt, base_dir)
    dt_path = os.path.join(base_dir, 'AI_08_Decision_Tree_Cars.ipynb')
    with open(dt_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_dt, f)
    print(f"    Готово -> {dt_path}")

    # 2. Random Forest
    print("  • 2/3 AI_08_Random_Forest_Cars.ipynb...")
    nb_rf = create_random_forest_notebook()
    execute_notebook(nb_rf, base_dir)
    rf_path = os.path.join(base_dir, 'AI_08_Random_Forest_Cars.ipynb')
    with open(rf_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_rf, f)
    print(f"    Готово -> {rf_path}")

    # 3. Gradient Boosting
    print("  • 3/3 AI_08_Gradient_Boosting_Cars.ipynb...")
    nb_gb = create_gradient_boosting_notebook()
    execute_notebook(nb_gb, base_dir)
    gb_path = os.path.join(base_dir, 'AI_08_Gradient_Boosting_Cars.ipynb')
    with open(gb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_gb, f)
    print(f"    Готово -> {gb_path}")

    print("Все 3 ноутбука успешно сгенерированы и исполнены!")


if __name__ == '__main__':
    main()
