"""
tasks/comparative_analysis.py
Модуль сравнительного анализа моделей: Linear Regression vs Decision Tree vs Random Forest vs Gradient Boosting
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def train_linear_regression_baseline(data_dict):
    """
    Обучает базовую линейную регрессию из AI 07 для прямого сопоставления.
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    preprocessor = data_dict['preprocessor']

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('lr', LinearRegression())
    ])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        'model_name': 'Linear Regression (AI 07)',
        'best_params': {'fit_intercept': True},
        'best_estimator': pipe,
        'fit_time_sec': fit_time,
        'train_metrics': {'MAE': train_mae, 'RMSE': train_rmse, 'R2': train_r2},
        'test_metrics': {'MAE': test_mae, 'MSE': test_mse, 'RMSE': test_rmse, 'R2': test_r2},
        'y_test_pred': y_test_pred
    }


def run_comparative_analysis(models_results, new_cars_df, output_dir="output"):
    """
    Формирует детальную сводную таблицу сравнения моделей,
    таблицу прогнозов для новых авто и графики сопоставления.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    interpretability_map = {
        'Linear Regression (AI 07)': 'Очень высокая (веса признаков)',
        'Decision Tree': 'Высокая (дерево условий if/else)',
        'Random Forest': 'Средняя (важность признаков / ансамбль)',
        'Gradient Boosting': 'Средняя/Низкая (последовательный бустинг)'
    }

    for res in models_results:
        name = res['model_name']
        tr = res['train_metrics']
        te = res['test_metrics']
        gap = tr['R2'] - te['R2']

        summary_rows.append({
            'Модель': name,
            'Train R²': round(tr['R2'], 4),
            'Test R²': round(te['R2'], 4),
            'Overfitting Gap (Δ R²)': round(gap, 4),
            'Test MAE (AZN)': round(te['MAE'], 2),
            'Test RMSE (AZN)': round(te['RMSE'], 2),
            'Время обучения (сек)': round(res['fit_time_sec'], 3),
            'Интерпретируемость': interpretability_map.get(name, 'Средняя')
        })

    summary_df = pd.DataFrame(summary_rows)

    # Прогнозы для 3 новых автомобилей
    preds_data = []
    for i, row in new_cars_df.iterrows():
        car_name = f"Авто {i+1}: {row['brand']} ({row['age_years']} лет, {row['mileage_km']:,} км)"
        preds_dict = {'Автомобиль': car_name}
        for res in models_results:
            model = res['best_estimator']
            pred_val = model.predict(new_cars_df.iloc[[i]])[0]
            preds_dict[res['model_name']] = round(pred_val, 2)
        preds_data.append(preds_dict)

    preds_df = pd.DataFrame(preds_data)

    # 1. Построение графика сравнения метрик
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = [r['Модель'] for r in summary_rows]
    test_r2_vals = [r['Test R²'] for r in summary_rows]
    test_rmse_vals = [r['Test RMSE (AZN)'] for r in summary_rows]

    colors = ['#455a64', '#2e7d32', '#1565c0', '#e65100']

    # Test R2
    axes[0].bar(model_names, test_r2_vals, color=colors, edgecolor='black', width=0.55)
    axes[0].set_title('Коэффициент детерминации (Test R²)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('R² score')
    axes[0].set_ylim(min(test_r2_vals) - 0.05, 1.0)
    for idx, v in enumerate(test_r2_vals):
        axes[0].text(idx, v + 0.005, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
    axes[0].grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[0].tick_params(axis='x', rotation=15)

    # Test RMSE
    axes[1].bar(model_names, test_rmse_vals, color=colors, edgecolor='black', width=0.55)
    axes[1].set_title('Ошибка прогноза (Test RMSE, AZN)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE (AZN)')
    for idx, v in enumerate(test_rmse_vals):
        axes[1].text(idx, v + 50, f"{v:.1f} AZN", ha='center', va='bottom', fontweight='bold')
    axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()

    # 2. Построение графика сравнения прогнозов
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(new_cars_df))
    width = 0.2

    for idx, res in enumerate(models_results):
        model_name = res['model_name']
        vals = [preds_df.loc[i, model_name] for i in range(len(new_cars_df))]
        offset = (idx - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=model_name, color=colors[idx], edgecolor='black')
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    car_labels = [
        f"Авто #1\nToyota (5 лет, 75k км)",
        f"Авто #2\nHyundai (10 лет, 180k км, ДТП)",
        f"Авто #3\nBMW (3 года, 40k км)"
    ]
    ax.set_ylabel('Прогноз стоимости (AZN)', fontsize=11)
    ax.set_title('Сравнение прогнозов стоимости для 3 новых авто разными моделями', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(car_labels, fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.set_ylim(0, max([preds_df[r['model_name']].max() for r in models_results]) * 1.18)

    plt.tight_layout()
    preds_plot_path = os.path.join(output_dir, 'predictions_comparison.png')
    plt.savefig(preds_plot_path, dpi=300)
    plt.close()

    return {
        'summary_table': summary_df,
        'predictions_table': preds_df,
        'metrics_plot_path': metrics_plot_path,
        'preds_plot_path': preds_plot_path
    }
