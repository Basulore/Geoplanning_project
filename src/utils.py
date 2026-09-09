"""
Основной модуль геопланирования. Содержит класс GeoPlanner.
"""
import pandas as pd
import numpy as np
from clustering import cluster_with_size_limit, build_route
from routing import distribute_visits


class GeoPlanner:
    """
    Класс для геопланирования и кластеризации точек посещения.
    """

    def __init__(self, eps=0.5, min_samples=2, max_cluster_size=10, random_seed=42):
        """
        Инициализация планировщика.

        Args:
            eps (float): Радиус кластеризации в км.
            min_samples (int): Минимальное количество точек в кластере.
            max_cluster_size (int): Максимальный размер кластера.
            random_seed (int): Seed для воспроизводимости.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_size = max_cluster_size
        self.random_seed = random_seed
        self.results = None
        self.statistics = None

    def load_data(self, filepath):
        """
        Загрузка и первичная очистка данных из CSV.

        Args:
            filepath (str): Путь к CSV файлу.

        Returns:
            pd.DataFrame: Загруженные и очищенные данные.
        """
        try:
            df = pd.read_csv(filepath)
            
            # Проверка и переименование колонок
            required_cols = {
                'point_id': ['point_id'],
                'lat': ['lat', 'latitude'],
                'lon': ['lon', 'longitude'],
                'visits_per_month': ['visits_per_month', 'n_visits'],
                'manager': ['manager']
            }
            
            for target, sources in required_cols.items():
                if target not in df.columns:
                    found = False
                    for src in sources:
                        if src in df.columns and src != target:
                            df[target] = df[src]
                            found = True
                            break
                    if not found and target != 'manager': # manager может отсутствовать
                        raise ValueError(f"Обязательная колонка '{target}' не найдена.")
                    elif target == 'manager' and not found:
                        df['manager'] = 0 # Значение по умолчанию
            
            # Очистка
            df = self._clean_data(df)
            return df
            
        except Exception as e:
            raise Exception(f"Ошибка загрузки данных: {e}")

    def _clean_data(self, df):
        """
        Валидация и очистка данных.

        Args:
            df (pd.DataFrame): Исходный DataFrame.

        Returns:
            pd.DataFrame: Очищенный DataFrame.
        """
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['lat', 'lon', 'point_id'])
        
        # Проверка координат
        invalid_coords = 0
        for idx, row in df_clean.iterrows():
            lat, lon = row['lat'], row['lon']
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                df_clean.loc[idx, ['lat', 'lon']] = np.nan
                invalid_coords += 1
        
        if invalid_coords > 0:
            df_clean = df_clean.dropna(subset=['lat', 'lon'])
            print(f"Удалено {invalid_coords} строк с некорректными координатами.")
        
        # Валидация visits_per_month
        if 'visits_per_month' in df_clean.columns:
            df_clean['visits_per_month'] = df_clean['visits_per_month'].fillna(1)
            df_clean['visits_per_month'] = df_clean['visits_per_month'].clip(0, 30)
            df_clean['visits_per_month'] = df_clean['visits_per_month'].astype(int)
        
        return df_clean

    def process(self, df):
        """
        Основной пайплайн обработки данных.

        Args:
            df (pd.DataFrame): Входные данные.

        Returns:
            pd.DataFrame: Результаты обработки.
        """
        # 1. Распределение по дням
        visits_df = distribute_visits(df, random_seed=self.random_seed)
        if len(visits_df) == 0:
            print("Нет данных для обработки.")
            return pd.DataFrame()

        # 2. Кластеризация и построение маршрутов для каждого дня
        all_results = []
        for day in sorted(visits_df['visit_day'].unique()):
            day_points = visits_df[visits_df['visit_day'] == day]
            
            # Кластеризация
            labels = cluster_with_size_limit(
                day_points,
                eps=self.eps,
                min_samples=self.min_samples,
                max_cluster_size=self.max_cluster_size
            )
            
            # Обработка каждого кластера (включая шум)
            unique_labels = np.unique(labels)
            for cluster_id in unique_labels:
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_points = day_points.iloc[cluster_indices]
                
                if cluster_id == -1:
                    # Шумовые точки не имеют маршрута
                    for idx in cluster_indices:
                        row = day_points.iloc[idx]
                        all_results.append({
                            'point_id': row['point_id'],
                            'visit_day': int(day),
                            'cluster_id': -1,
                            'order_in_route': None
                        })
                else:
                    # Построение маршрута для кластера
                    route_order = build_route(cluster_points)
                    for order, idx in enumerate(route_order):
                        row = cluster_points.iloc[idx]
                        all_results.append({
                            'point_id': row['point_id'],
                            'visit_day': int(day),
                            'cluster_id': int(cluster_id),
                            'order_in_route': order + 1
                        })

        self.results = pd.DataFrame(all_results)
        self._add_statistics()
        return self.results

    def _add_statistics(self):
        """Сбор и вывод статистики."""
        if self.results is None or len(self.results) == 0:
            return

        stats = {
            'total_visits': len(self.results),
            'unique_points': len(self.results['point_id'].unique()),
            'unique_days': len(self.results['visit_day'].unique()),
            'clusters_count': len(self.results[self.results['cluster_id'] != -1]['cluster_id'].unique()),
            'noise_points': len(self.results[self.results['cluster_id'] == -1])
        }
        self.statistics = stats
        
        print("\n=== Статистика обработки ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

    def save_results(self, filepath):
        """Сохранение результатов в CSV."""
        if self.results is None:
            raise ValueError("Нет результатов для сохранения.")
        self.results.to_csv(filepath, index=False)
        print(f"Результаты сохранены в {filepath}")

    def get_summary(self):
        """Возвращает текстовую сводку по дням."""
        if self.results is None:
            return "Нет результатов."

        summary = ["=== Сводка по дням ==="]
        for day in sorted(self.results['visit_day'].unique()):
            day_data = self.results[self.results['visit_day'] == day]
            clusters = day_data[day_data['cluster_id'] != -1]
            noise = day_data[day_data['cluster_id'] == -1]
            
            summary.append(f"\nДень {day}:")
            summary.append(f"  Всего точек: {len(day_data)}")
            summary.append(f"  В кластерах: {len(clusters)}")
            summary.append(f"  Шум: {len(noise)}")
            summary.append(f"  Количество кластеров: {len(clusters['cluster_id'].unique())}")
            
            cluster_sizes = clusters.groupby('cluster_id').size().tolist()
            summary.append(f"  Размеры кластеров: {cluster_sizes}")

        return "\n".join(summary)
