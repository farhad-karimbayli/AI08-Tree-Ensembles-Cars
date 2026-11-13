"""
main.py
Главный исполняемый скрипт для проекта AI 08:
Деревья решений, Случайный лес, Градиентный бустинг и сравнительный анализ.
"""

import sys
import os

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tasks.data_loader import load_and_preprocess_data, get_new_cars_df
from tasks.decision_tree_model import train_decision_tree
from tasks.random_forest_model import train_random_forest
from tasks.gradient_boosting_model import train_gradient_boosting
from tasks.comparative_analysis import train_linear_regression_baseline, run_comparative_analysis


def main():
    print("=" * 80)
    print(" AI 08: РЕГРЕССИЯ СТОИМОСТИ АВТОМОБИЛЕЙ НА БАЗЕ ДРЕВОВИДНЫХ АНСАМБЛЕЙ")
    print(" (Decision Tree, Random Forest, Gradient Boosting + Сравнительный анализ)")
    print("=" * 80)

    # 1. Загрузка данных
    print("\n[1/5] Загрузка данных и подготовка признаков...")
    data_dict = load_and_preprocess_data()
    new_cars_df = get_new_cars_df()
    print(f"  • Обучающая выборка: {data_dict['X_train'].shape[0]} записей")
    print(f"  • Тестовая выборка  : {data_dict['X_test'].shape[0]} записей")
    print(f"  • Числовые признаки: {data_dict['numeric_features']}")
    print(f"  • Категориальные   : {data_dict['categorical_features']}")

    # 2. Базовая Linear Regression (AI 07)
    print("\n[2/5] Обучение базовой модели Linear Regression (из AI 07)...")
    lr_res = train_linear_regression_baseline(data_dict)
    print(f"  Test R2: {lr_res['test_metrics']['R2']:.4f} | Test RMSE: {lr_res['test_metrics']['RMSE']:.2f} AZN | MAE: {lr_res['test_metrics']['MAE']:.2f} AZN")

    # 3. Decision Tree
    print("\n[3/5] Подбор гиперпараметров и обучение Decision Tree Regressor...")
    dt_res = train_decision_tree(data_dict)
    print(f"  Лучшие параметры: {dt_res['best_params']}")
    print(f"  Test R2: {dt_res['test_metrics']['R2']:.4f} | Test RMSE: {dt_res['test_metrics']['RMSE']:.2f} AZN | MAE: {dt_res['test_metrics']['MAE']:.2f} AZN")

    # 4. Random Forest
    print("\n[4/5] Подбор гиперпараметров и обучение Random Forest Regressor...")
    rf_res = train_random_forest(data_dict)
    print(f"  Лучшие параметры: {rf_res['best_params']}")
    print(f"  Test R2: {rf_res['test_metrics']['R2']:.4f} | Test RMSE: {rf_res['test_metrics']['RMSE']:.2f} AZN | MAE: {rf_res['test_metrics']['MAE']:.2f} AZN")

    # 5. Gradient Boosting
    print("\n[5/5] Подбор гиперпараметров и обучение Gradient Boosting Regressor...")
    gb_res = train_gradient_boosting(data_dict)
    print(f"  Лучшие параметры: {gb_res['best_params']}")
    print(f"  Test R2: {gb_res['test_metrics']['R2']:.4f} | Test RMSE: {gb_res['test_metrics']['RMSE']:.2f} AZN | MAE: {gb_res['test_metrics']['MAE']:.2f} AZN")

    # 6. Сравнительный анализ
    print("\n" + "=" * 80)
    print(" СВОДНЫЙ СРАВНИТЕЛЬНЫЙ АНАЛИЗ ВСЕХ АЛГОРИТМОВ")
    print("=" * 80)
    all_models = [lr_res, dt_res, rf_res, gb_res]
    comp_res = run_comparative_analysis(all_models, new_cars_df)

    print("\n--- 1. Сравнение метрик качества на выборках ---")
    print(comp_res['summary_table'].to_string(index=False))

    print("\n--- 2. Сравнение прогнозов стоимости для 3 новых автомобилей (AZN) ---")
    print(comp_res['predictions_table'].to_string(index=False))

    print("\nГрафики успешно сохранены в папку 'output/':")
    print("  • output/decision_tree_results.png")
    print("  • output/random_forest_results.png")
    print("  • output/gradient_boosting_results.png")
    print("  • output/metrics_comparison.png")
    print("  • output/predictions_comparison.png")
    print("=" * 80)
    print(" Выполнение завершено успешно!")
    print("=" * 80)


if __name__ == '__main__':
    main()
