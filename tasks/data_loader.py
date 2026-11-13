"""
tasks/data_loader.py
Модуль загрузки данных, подготовки признаков и построения ColumnTransformer
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def get_data_path() -> str:
    """Определяет путь к файлу Used_Cars_Homework.csv."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "Used_Cars_Homework.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "Used_Cars_Homework.csv"),
        "Used_Cars_Homework.csv",
        "data/Used_Cars_Homework.csv"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Файл Used_Cars_Homework.csv не найден!")


def load_and_preprocess_data():
    """
    Загружает датасет, вычисляет дополнительный признак mileage_per_year,
    разделяет признаки и целевую переменную, делит на train и test выборки.
    """
    path = get_data_path()
    df = pd.read_csv(path)

    # Генерация признака среднегодового пробега
    df['mileage_per_year'] = df['mileage_km'] / df['age_years']

    target_col = 'price_azn'
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    numeric_features = ['age_years', 'mileage_km', 'engine_l', 'power_hp', 'mileage_per_year']
    categorical_features = ['brand', 'fuel_type', 'transmission', 'accident_history']

    # Пайплайн предобработки (в деревьях решений масштабирование StandardScaler не требуется,
    # так как разбиения зависят только от монотонного порядка значений признака)
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'preprocessor': preprocessor
    }


def get_new_cars_df() -> pd.DataFrame:
    """
    Возвращает DataFrame с 3 новыми автомобилями для оценки стоимости
    (из задания AI 07 / AI 08).
    """
    new_cars = [
        {
            'brand': 'Toyota',
            'age_years': 5,
            'mileage_km': 75000,
            'engine_l': 1.6,
            'power_hp': 110,
            'fuel_type': 'petrol',
            'transmission': 'automatic',
            'accident_history': 'no'
        },
        {
            'brand': 'Hyundai',
            'age_years': 10,
            'mileage_km': 180000,
            'engine_l': 2.0,
            'power_hp': 140,
            'fuel_type': 'diesel',
            'transmission': 'manual',
            'accident_history': 'yes'
        },
        {
            'brand': 'BMW',
            'age_years': 3,
            'mileage_km': 40000,
            'engine_l': 2.5,
            'power_hp': 170,
            'fuel_type': 'petrol',
            'transmission': 'automatic',
            'accident_history': 'no'
        }
    ]
    df_new = pd.DataFrame(new_cars)
    df_new['mileage_per_year'] = df_new['mileage_km'] / df_new['age_years']
    return df_new
