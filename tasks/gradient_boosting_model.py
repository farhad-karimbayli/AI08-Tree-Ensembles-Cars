"""
tasks/gradient_boosting_model.py
Модель Gradient Boosting (Градиентный бустинг) для регрессии стоимости автомобилей
"""

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def train_gradient_boosting(data_dict, output_dir="output"):
    """
    Обучает GradientBoostingRegressor с подбором гиперпараметров через GridSearchCV.
    Оценивает качество, строит графики и извлекает важность признаков.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    preprocessor = data_dict['preprocessor']

    gbr = GradientBoostingRegressor(random_state=42)

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
    grid_search = GridSearchCV(
        pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    fit_time = time.time() - t0

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Предсказания
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Метрики
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Извлечение важности признаков
    gbr_step = best_model.named_steps['gbr']
    prep_step = best_model.named_steps['preprocessor']
    feature_names = prep_step.get_feature_names_out()
    clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    importances = gbr_step.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. True vs Predicted
    axes[0].scatter(y_test, y_test_pred, color='#e65100', alpha=0.75, edgecolors='k', s=45)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальный прогноз (y=x)')
    axes[0].set_title(f'Gradient Boosting: Реальные vs Прогнозные\n(Test R2 = {test_r2:.4f}, RMSE = {test_rmse:.1f} AZN)', fontsize=11)
    axes[0].set_xlabel('Реальная стоимость (True Values, AZN)')
    axes[0].set_ylabel('Прогноз модели (Predicted Values, AZN)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # 2. Важность признаков
    top_feats = feat_imp_df.head(8).sort_values('Importance', ascending=True)
    axes[1].barh(top_feats['Feature'], top_feats['Importance'], color='#f57c00', edgecolor='black')
    axes[1].set_title('Gradient Boosting: Топ-8 важнейших признаков', fontsize=11)
    axes[1].set_xlabel('Важность (Feature Importance)')
    axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'gradient_boosting_results.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return {
        'model_name': 'Gradient Boosting',
        'best_params': best_params,
        'best_estimator': best_model,
        'fit_time_sec': fit_time,
        'train_metrics': {'MAE': train_mae, 'RMSE': train_rmse, 'R2': train_r2},
        'test_metrics': {'MAE': test_mae, 'MSE': test_mse, 'RMSE': test_rmse, 'R2': test_r2},
        'feature_importances': feat_imp_df,
        'plot_path': plot_path,
        'y_test_pred': y_test_pred
    }
