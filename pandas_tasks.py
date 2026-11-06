"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """

    df = pd.read_csv(data.csv_path)#загрузка CSV
    row_count = int(len(df))# определение кол-ва строк

    # Подсчет пропусков
    missing_by_column = df.isna().sum().to_dict()
    missing_by_column = {str(k): int(v) for k, v in missing_by_column.items()}
    # Кол-во пассажиров >30
    adults_over_30_count = int((df['Age'] > 30).sum())
    # Средний возраст для каждого Pclass
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age_by_pclass = {int(k): float(v) for k, v in mean_age_by_pclass.items()}
    # Доля выживших
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival_rate_by_pclass = {int(k): float(v) for k, v in survival_rate_by_pclass.items()}
    # Пять лучших результатов Fare в порядке убывания
    highest_fares = df['Fare'].nlargest(5).tolist()
    highest_fares = [float(x) for x in highest_fares]
    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )
if __name__ == '__main__':
    from grader_contracts.pandas_tasks import TitanicInput
    
    test_input = TitanicInput(csv_path='titanic.csv')
    result = analyze_titanic(test_input)
    
    print(f"Строк: {result.row_count}")
    print(f"Пропуски: {result.missing_by_column}")
    print(f"Старше 30: {result.adults_over_30_count}")
    print(f"Средний возраст: {result.mean_age_by_pclass}")
    print(f"Выживаемость: {result.survival_rate_by_pclass}")
    print(f"Топ-5 тарифов: {result.highest_fares}")