"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd
from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    # Загружаем таблицу с табуляцией, обрабатываем NA
    df = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    
    # Убираем пробелы по краям названий колонок
    df.columns = df.columns.str.strip()
    
    # Определяем колонку пола и нормализуем значения
    gender_col = 'Gender'
    df[gender_col] = df[gender_col].astype(str).str.strip()
    
    # Учитываем возможные варианты написания пола (M/F, Male/Female, и т.д.)
    male_mask = df[gender_col].isin(['M', 'Male', 'мужской', 'Мужской', 'Мужчина', 'Мужчины'])
    female_mask = df[gender_col].isin(['F', 'Female', 'женский', 'Женский', 'Женщина', 'Женщины'])
    
    male_df = df[male_mask].copy()
    female_df = df[female_mask].copy()
    
    # Считаем количество
    men_count = int(len(male_df))
    women_count = int(len(female_df))
    
    # Признаки для корреляции (английские названия, как в файле)
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    target = 'MRI_Count'
    
    # Функция для расчёта корреляций
    def calc_correlations(sub_df):
        correlations = {}
        for feat in features:
            # Берём только строки, где оба значения не NaN
            valid = sub_df[[feat, target]].dropna()
            if len(valid) > 1:
                corr = valid[feat].corr(valid[target])
                correlations[feat] = float(corr) if pd.notna(corr) else 0.0
            else:
                correlations[feat] = 0.0
        return correlations
    
    # Считаем корреляции для мужчин и женщин
    men_mri_correlation = calc_correlations(male_df)
    women_mri_correlation = calc_correlations(female_df)
    
    # Находим признак с максимальным модулем корреляции среди обеих групп
    all_abs_correlations = {}
    for feat in features:
        all_abs_correlations[feat] = max(
            abs(men_mri_correlation.get(feat, 0.0)),
            abs(women_mri_correlation.get(feat, 0.0))
        )
    
    strongest_mri_feature = max(all_abs_correlations, key=all_abs_correlations.get)
    
    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
    )
if __name__ == '__main__':

    from grader_contracts.correlation_tasks import BrainDataInput   
    try:
        test_input = BrainDataInput(csv_path='brainsize.txt')
        result = analyze_brain_correlations(test_input)
        
        print(f" Мужчин в выборке: {result.men_count}")
        print(f" Женщин в выборке: {result.women_count}")
        print(" Корреляции с MRI_Count (Мужчины):")
        for feat, corr in result.men_mri_correlation.items():
            print(f"   {feat}: {corr:.4f}")
            
        print(" Корреляции с MRI_Count (Женщины):")
        for feat, corr in result.women_mri_correlation.items():
            print(f"   {feat}: {corr:.4f}")
            
        print(f"Сильнейший предиктор MRI_Count : {result.strongest_mri_feature}")     
    except Exception as e:
        print(f" Ошибка при выполнении: {e}")