"""
Вспомогательные функции для визуализации и логирования.
"""
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from clustering import calculate_distance


def visualize_map(points_df, results_df=None, title="Геопланирование"):
    """
    Создает интерактивную карту с кластерами и маршрутами.

    Args:
        points_df (pd.DataFrame): DataFrame с точками (lat, lon, point_id).
        results_df (pd.DataFrame): DataFrame с результатами кластеризации.
        title (str): Заголовок карты.

    Returns:
        folium.Map: Интерактивная карта.
    """
    if len(points_df) == 0:
        raise ValueError("Нет данных для визуализации.")

    center_lat = points_df['lat'].mean()
    center_lon = points_df['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

    if results_df is not None:
        # Объединяем данные
        merged = points_df.merge(results_df[['point_id', 'cluster_id', 'visit_day', 'order_in_route']], 
                                 on='point_id', how='left')
        
        # Заполняем пропуски
        merged['cluster_id'] = merged['cluster_id'].fillna(-1).astype(int)
        merged['order_in_route'] = merged['order_in_route'].fillna(0).astype(int)
        
        clusters = merged[merged['cluster_id'] != -1]['cluster_id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(clusters))))
        color_map = {cluster: mcolors.rgb2hex(colors[i % len(colors)]) 
                     for i, cluster in enumerate(clusters)}
        color_map[-1] = '#808080'  # Цвет для шума

        # Рисуем линии маршрутов (только для точек с order_in_route > 0)
        for day in merged['visit_day'].unique():
            day_data = merged[merged['visit_day'] == day]
            for cluster_id in clusters:
                cluster_data = day_data[day_data['cluster_id'] == cluster_id].copy()
                cluster_data = cluster_data[cluster_data['order_in_route'] > 0].sort_values('order_in_route')
                
                if len(cluster_data) > 1:
                    points = cluster_data[['lat', 'lon']].values.tolist()
                    folium.PolyLine(
                        points,
                        color=color_map[cluster_id],
                        weight=2.5,
                        opacity=0.8,
                        popup=f"Маршрут {cluster_id} (День {day})"
                    ).add_to(m)

        # Рисуем точки
        for _, row in merged.iterrows():
            cluster_id = row['cluster_id']
            color = color_map.get(cluster_id, '#0000FF')
            
            popup_text = f"{row['point_id']}<br>День: {row['visit_day']}<br>Кластер: {cluster_id}"
            if cluster_id != -1 and row['order_in_route'] > 0:
                popup_text += f"<br>Порядок: {row['order_in_route']}"
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6 if cluster_id != -1 else 3,
                popup=popup_text,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7 if cluster_id != -1 else 0.3
            ).add_to(m)

        # Легенда
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: auto; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white; padding: 10px; border-radius: 5px;">
        <p><strong>Легенда</strong></p>
        '''
        for cluster, color in list(color_map.items())[:10]:
            if cluster == -1:
                legend_html += f'<p><span style="color:{color};">●</span> Шум</p>'
            else:
                legend_html += f'<p><span style="color:{color};">●</span> Кластер {cluster}</p>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

    else:
        # Простая карта без кластеров
        for _, row in points_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                popup=row['point_id'],
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6
            ).add_to(m)

    return m


def plot_cluster_analysis(results_df):
    """
    Визуализация статистики кластеризации.

    Args:
        results_df (pd.DataFrame): Результаты кластеризации.
    """
    if results_df is None or len(results_df) == 0:
        print("Нет данных для анализа.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Распределение визитов по дням
    day_counts = results_df['visit_day'].value_counts().sort_index()
    axes[0, 0].bar(day_counts.index, day_counts.values)
    axes[0, 0].set_xlabel('День месяца')
    axes[0, 0].set_ylabel('Количество визитов')
    axes[0, 0].set_title('Распределение визитов по дням')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Размеры кластеров
    clusters = results_df[results_df['cluster_id'] != -1]
    cluster_sizes = clusters.groupby(['visit_day', 'cluster_id']).size().reset_index(name='size')
    axes[0, 1].hist(cluster_sizes['size'], bins=range(1, 15), alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Размер кластера')
    axes[0, 1].set_ylabel('Количество кластеров')
    axes[0, 1].set_title('Распределение размеров кластеров')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Количество кластеров по дням
    clusters_per_day = results_df[results_df['cluster_id'] != -1].groupby('visit_day')['cluster_id'].nunique()
    axes[1, 0].bar(clusters_per_day.index, clusters_per_day.values)
    axes[1, 0].set_xlabel('День месяца')
    axes[1, 0].set_ylabel('Количество кластеров')
    axes[1, 0].set_title('Количество кластеров по дням')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Соотношение кластеров и шума
    cluster_count = len(results_df[results_df['cluster_id'] != -1])
    noise_count = len(results_df[results_df['cluster_id'] == -1])
    axes[1, 1].pie([cluster_count, noise_count], 
                   labels=['В кластерах', 'Шум'], 
                   autopct='%1.1f%%', 
                   colors=['#4CAF50', '#FFA07A'])
    axes[1, 1].set_title('Соотношение кластеров и шума')
    
    plt.tight_layout()
    plt.show()

