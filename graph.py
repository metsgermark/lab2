"""Визуализация для отчёта по лабораторной работе №2."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
sns.set_theme(style="whitegrid")

def create_visualizations():
    # Загрузка данных
    df = pd.read_csv('brainsize.txt', sep='\t', na_values=['NA'])
    df.columns = df.columns.str.strip()
    
    # Нормализация пола
    df['Gender'] = df['Gender'].astype(str).str.strip()
    male_mask = df['Gender'].isin(['M', 'Male', 'мужской', 'Мужской', 'Мужчина'])
    female_mask = df['Gender'].isin(['F', 'Female', 'женский', 'Женский', 'Женщина'])
    
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    target = 'MRI_Count'
    
    
    # ГРАФИК 1: Тепловые карты корреляций
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Матрицы корреляций Пирсона с MRI_Count', fontsize=16, fontweight='bold')
    
    all_cols = features + [target]
    
    # Мужчины
    male_corr = df[male_mask][all_cols].corr()
    sns.heatmap(male_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0], fmt='.2f', vmin=-1, vmax=1)
    axes[0].set_title('Мужчины', fontsize=14)
    
    # Женщины
    female_corr = df[female_mask][all_cols].corr()
    sns.heatmap(female_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1], fmt='.2f', vmin=-1, vmax=1)
    axes[1].set_title('Женщины', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    print(" Сохранено: correlation_heatmaps.png")
    plt.show()
    
    
    # ГРАФИК 2: Точечный график для сильнейшего признака
    
    # Находим сильнейший признак динамически
    best_feature = None
    max_corr = 0
    for feat in features:
        c_m = abs(df[male_mask][[feat, target]].corr().iloc[0, 1])
        c_f = abs(df[female_mask][[feat, target]].corr().iloc[0, 1])
        if max(c_m, c_f) > max_corr:
            max_corr = max(c_m, c_f)
            best_feature = feat
            
    print(f" Сильнейший признак для графика: {best_feature} (модуль корреляции: {max_corr:.3f})")
    
    plt.figure(figsize=(8, 6))
    
    male_data = df[male_mask].dropna(subset=[best_feature, target])
    female_data = df[female_mask].dropna(subset=[best_feature, target])
    
    plt.scatter(male_data[best_feature], male_data[target], 
                color='royalblue', alpha=0.7, label=f'Мужчины (n={len(male_data)})', edgecolors='white')
    plt.scatter(female_data[best_feature], female_data[target], 
                color='crimson', alpha=0.7, label=f'Женщины (n={len(female_data)})', edgecolors='white')
    
    plt.title(f'Зависимость {target} от {best_feature}', fontsize=14, fontweight='bold')
    plt.xlabel(best_feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('scatter_plot_best_feature.png', dpi=300, bbox_inches='tight')
    print(" Сохранено: scatter_plot_best_feature.png")
    plt.show()

if __name__ == '__main__':
    create_visualizations()