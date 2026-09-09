"""
Основной скрипт для запуска из командной строки.
"""
import argparse
import sys
import os
import matplotlib.pyplot as plt

# Добавляем путь к src в sys.path, если скрипт запускается из корня проекта
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import GeoPlanner
from visualize import visualize_map, plot_cluster_analysis


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description='Геопланирование и кластеризация точек посещения'
    )
    parser.add_argument('input_file', type=str, help='Путь к входному CSV файлу')
    parser.add_argument('-o', '--output', type=str, default='results.csv', 
                        help='Путь для сохранения результатов (по умолчанию: results.csv)')
    parser.add_argument('-e', '--eps', type=float, default=0.5, 
                        help='Радиус кластеризации в км (по умолчанию: 0.5)')
    parser.add_argument('-m', '--min-samples', type=int, default=2, 
                        help='Минимальное количество точек в кластере (по умолчанию: 2)')
    parser.add_argument('-M', '--max-cluster-size', type=int, default=10, 
                        help='Максимальный размер кластера (по умолчанию: 10)')
    parser.add_argument('-r', '--random-seed', type=int, default=42, 
                        help='Seed для воспроизводимости (по умолчанию: 42)')
    parser.add_argument('--map', type=str, default='map.html', 
                        help='Сохранить карту в HTML файл')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять результаты')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Ошибка: файл {args.input_file} не найден.")
        sys.exit(1)

    try:
        planner = GeoPlanner(
            eps=args.eps,
            min_samples=args.min_samples,
            max_cluster_size=args.max_cluster_size,
            random_seed=args.random_seed
        )

        print(f"Загрузка данных из {args.input_file}...")
        data = planner.load_data(args.input_file)
        print(f"Загружено {len(data)} точек.")

        print("Обработка данных...")
        results = planner.process(data)
        if results.empty:
            print("Нет данных для обработки.")
            sys.exit(0)
            
        print(f"Обработано {len(results)} визитов.")
        print(planner.get_summary())

        if not args.no_save:
            planner.save_results(args.output)

        print("Создание карты...")
        try:
            map_obj = visualize_map(data, results)
            map_obj.save(args.map)
            print(f"Карта сохранена в {args.map}.")
        except Exception as e:
            print(f"Ошибка при создании карты: {e}")

        print("Создание графиков...")
        try:
            plot_cluster_analysis(results)
            plt.show()
        except Exception as e:
            print(f"Ошибка при создании графиков: {e}")

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

