"""
Модуль кластеризации для геопланирования.
Реализует кластеризацию с помощью DBSCAN и принудительное разбиение кластеров.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import warnings

warnings.filterwarnings('ignore')


def calculate_distance(point1, point2):
    """
    Расчет расстояния между двумя географическими точками в километрах.

    Args:
        point1 (tuple): (latitude, longitude)
        point2 (tuple): (latitude, longitude)

    Returns:
        float: расстояние в километрах.
    """
    try:
        return geodesic(point1, point2).kilometers
    except (ValueError, TypeError):
        return np.inf


def cluster_points(points, eps=0.5, min_samples=2):
    """
    Кластеризация точек с использованием DBSCAN.

    Args:
        points (np.ndarray): Массив координат [[lat, lon], ...].
        eps (float): Максимальное расстояние в км для формирования кластера.
        min_samples (int): Минимальное количество точек в кластере.

    Returns:
        np.ndarray: Метки кластеров (-1 для шума).
    """
    if len(points) == 0:
        return np.array([])

    coords_rad = np.radians(points)
    clustering = DBSCAN(
        eps=eps / 6371.0,
        min_samples=min_samples,
        metric='haversine'
    )
    return clustering.fit_predict(coords_rad)


def split_cluster(points_df, max_cluster_size):
    """
    Рекурсивно разбивает кластер, если его размер превышает лимит.

    Args:
        points_df (pd.DataFrame): DataFrame с точками одного кластера.
        max_cluster_size (int): Максимально допустимый размер кластера.

    Returns:
        np.ndarray: Массив меток для всех точек из points_df.
    """
    if len(points_df) <= max_cluster_size:
        return np.zeros(len(points_df))

    # Используем KMeans для разбиения большого кластера на две части
    n_clusters = max(2, len(points_df) // max_cluster_size)
    coords = points_df[['lat', 'lon']].values
    
    # Если координаты почти идентичны, KMeans может не сработать
    if np.std(coords, axis=0).sum() < 1e-6:
        # Если все точки в одном месте, просто делим их на части по порядку
        labels = np.array([i % n_clusters for i in range(len(points_df))])
        return labels

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    # Рекурсивно проверяем и разбиваем каждый новый кластер
    new_labels = np.zeros(len(points_df), dtype=int)
    offset = 0
    for cluster_id in np.unique(labels):
        sub_df = points_df.iloc[labels == cluster_id]
        sub_labels = split_cluster(sub_df, max_cluster_size)
        unique_sub_labels = np.unique(sub_labels)
        
        # Присваиваем уникальные ID для вложенных кластеров
        for i, sub_id in enumerate(unique_sub_labels):
            new_labels[labels == cluster_id] = (sub_labels == sub_id) * (offset + i) + offset
        offset += len(unique_sub_labels)
        
    return new_labels


def cluster_with_size_limit(points_df, eps=0.5, min_samples=2, max_cluster_size=10):
    """
    Кластеризация с ограничением максимального размера кластера.

    Args:
        points_df (pd.DataFrame): DataFrame с колонками lat, lon.
        eps (float): Радиус кластеризации.
        min_samples (int): Минимальное количество точек в кластере.
        max_cluster_size (int): Максимальный размер кластера.

    Returns:
        np.ndarray: Метки кластеров.
    """
    if len(points_df) == 0:
        return np.array([])

    points = points_df[['lat', 'lon']].values
    labels = cluster_points(points, eps=eps, min_samples=min_samples)
    
    # Разбираемся с кластерами, которые превышают лимит
    unique_labels = np.unique(labels)
    final_labels = np.zeros(len(points), dtype=int)
    offset = 0
    
    for label in unique_labels:
        if label == -1:
            # Шум остаётся шумом
            final_labels[labels == -1] = -1
            continue
        
        cluster_indices = np.where(labels == label)[0]
        cluster_df = points_df.iloc[cluster_indices].reset_index(drop=True)
        
        if len(cluster_df) > max_cluster_size:
            # Разбиваем большой кластер
            sub_labels = split_cluster(cluster_df, max_cluster_size)
            # Присваиваем новые уникальные ID
            for i, sub_id in enumerate(np.unique(sub_labels)):
                final_labels[cluster_indices[sub_labels == sub_id]] = offset + i
            offset += len(np.unique(sub_labels))
        else:
            # Кластер в порядке, сохраняем метку
            final_labels[cluster_indices] = offset
            offset += 1
    
    # Смещаем метки так, чтобы первый кластер был 0, второй - 1, и т.д.
    # Но оставляем шум как -1
    unique_final = np.unique(final_labels)
    mapping = {old: new for new, old in enumerate(unique_final) if old != -1}
    if -1 in mapping: del mapping[-1]
    
    final_labels_mapped = np.array([mapping.get(label, -1) for label in final_labels])
    
    return final_labels_mapped


def build_route(points_df):
    """
    Построение маршрута методом ближайшего соседа.

    Args:
        points_df (pd.DataFrame): DataFrame с точками кластера.

    Returns:
        list: Порядок индексов точек в маршруте.
    """
    if len(points_df) <= 1:
        return list(range(len(points_df)))

    points = points_df[['lat', 'lon']].values
    n_points = len(points)

    # Стартуем с точки, ближайшей к центру кластера
    centroid = points.mean(axis=0)
    start_idx = np.argmin([calculate_distance(point, centroid) for point in points])

    route = [start_idx]
    unvisited = set(range(n_points))
    unvisited.remove(start_idx)

    while unvisited:
        current = route[-1]
        # Находим ближайшую непосещенную точку
        distances = []
        for idx in unvisited:
            dist = calculate_distance(points[current], points[idx])
            distances.append((idx, dist))
        
        if distances:
            nearest = min(distances, key=lambda x: x[1])
            route.append(nearest[0])
            unvisited.remove(nearest[0])
        else:
            break

    return route


