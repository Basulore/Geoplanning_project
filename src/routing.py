"""
Модуль для равномерного распределения визитов по дням месяца.
"""
import numpy as np
import pandas as pd


def distribute_visits(df, random_seed=42):
    """
    Равномерно распределяет N визитов по 30 дням месяца.

    Алгоритм: месяц разбивается на N равных интервалов, и в каждом интервале
    случайным образом выбирается один день. Это гарантирует, что дни будут
    равномерно распределены по всему месяцу.

    Args:
        df (pd.DataFrame): DataFrame с колонками point_id, lat, lon, visits_per_month, manager.
        random_seed (int): Seed для воспроизводимости.

    Returns:
        pd.DataFrame: DataFrame с колонками point_id, lat, lon, visit_day, manager.
    """
    np.random.seed(random_seed)
    visits = []

    for _, row in df.iterrows():
        n_visits = int(row['visits_per_month'])
        if n_visits <= 0:
            continue
        if n_visits > 30:
            n_visits = 30

        if n_visits == 1:
            # Случайный день в месяце
            day = np.random.randint(1, 31)
            days = [day]
        else:
            # Разбиваем месяц на n_visits равных интервалов
            # Например, для n_visits=2: интервалы [1-15] и [16-30]
            # для n_visits=3: [1-10], [11-20], [21-30] и т.д.
            edges = np.linspace(0, 30, n_visits + 1, dtype=int)
            days = []
            for i in range(n_visits):
                low = edges[i] + 1
                high = edges[i + 1]
                # Гарантируем, что low <= high
                if low > high:
                    low, high = high, low
                # Выбираем случайный день в интервале
                day = np.random.randint(low, high + 1)
                days.append(day)
            
            # Убираем дубликаты (если интервалы маленькие)
            days = sorted(set(days))
            
            # Если дубликаты привели к уменьшению количества дней,
            # добавляем случайные недостающие дни
            while len(days) < n_visits:
                new_day = np.random.randint(1, 31)
                if new_day not in days:
                    days.append(new_day)
            days = sorted(days)
        
        for day in days:
            visits.append({
                'point_id': row['point_id'],
                'lat': row['lat'],
                'lon': row['lon'],
                'visit_day': int(day),
                'manager': int(row['manager']) if pd.notna(row['manager']) else 0
            })

    return pd.DataFrame(visits)
