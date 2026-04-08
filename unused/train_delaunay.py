import os
import json
import numpy as np
import cv2
import pickle
from scipy.spatial import Delaunay


def collect_points(root_dir, _split = ['train']):  # Мутабельный тип, но внутри функции изменяться не будет
    points_data = {"top": [], "bottom": []}
    for split in _split:  # for split in ['train', 'val']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir): continue
        for session in os.listdir(split_dir):
            sess_path = os.path.join(split_dir, session)
            if not os.path.isdir(sess_path): continue
            for cam_type in ['top', 'bottom']:
                json_path = os.path.join(sess_path, f'coords_{cam_type}.json')
                if not os.path.exists(json_path): continue
                with open(json_path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    c1 = item.get('image1_coordinates', [])
                    c2 = item.get('image2_coordinates', [])
                    if len(c1) != len(c2): continue
                    for p_dst, p_src in zip(c1, c2):
                        points_data[cam_type].append((
                            np.array([p_src['x'], p_src['y']], dtype=np.float32),
                            np.array([p_dst['x'], p_dst['y']], dtype=np.float32)
                        ))
    return points_data


def evaluate_delaunay(mapper_data, val_pairs):
    """Вычисляет MED на валидации"""
    if mapper_data is None or not val_pairs:
        return float('inf')

    tri = mapper_data["triangulation"]
    transforms = mapper_data["transforms"]
    pts_src = mapper_data["pts_src"]

    errors = []

    for src_pt, dst_pt_true in val_pairs:
        simplex_idx = tri.find_simplex(src_pt.reshape(1, -1))[0]

        if simplex_idx >= 0:
            # Применяем аффинное преобразование
            M = transforms[simplex_idx]
            src_pt_homo = np.append(src_pt, 1.0).reshape(3, 1)
            dst_pt_pred = (M @ src_pt_homo).flatten()

            # Вычисляем евклидово расстояние
            error = np.sqrt(np.sum((dst_pt_pred - dst_pt_true) ** 2))
            errors.append(error)

    return np.mean(errors) if errors else float('inf')


def build_delaunay_mapper(points_pairs):
    if not points_pairs:
        return None

    pts_src = np.array([p[0] for p in points_pairs])
    pts_dst = np.array([p[1] for p in points_pairs])

    print(f"Считается триангуляция для {len(pts_src)} точек")

    tri = Delaunay(pts_src)

    # Для каждого треугольника вычисляем аффинную матрицу (2x3)
    transforms = {}

    for i, simplex in enumerate(tri.simplices):
        # 3 вершины треугольника
        src_tri = pts_src[simplex]
        dst_tri = pts_dst[simplex]

        M = cv2.getAffineTransform(src_tri, dst_tri)
        transforms[i] = M

    return {
        "triangulation": tri,
        "transforms": transforms,
        "pts_src": pts_src,
        "pts_dst": pts_dst
    }


if __name__ == "__main__":
    ROOT_DIR = '../coord_data'

    points_train = collect_points(ROOT_DIR, _split = ['train'])
    points_val = collect_points(ROOT_DIR, _split = ['val'])
    results = {}

    for cam in ["top", "bottom"]:
        print(f"\nОбучаем для {cam}...")
        mapper_data = build_delaunay_mapper(points_train[cam])

        if mapper_data:
            save_path = f'../models/delaunay_{cam}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(mapper_data, f)
            print(f"Сохранены данные о {len(mapper_data['transforms'])} треугольниках в {save_path}")

            # Валидация
            med = evaluate_delaunay(mapper_data, points_val[cam])
            results[f'{cam} > door2'] = med
            print(f"Валидационный MED ({cam} > door2): {med:.2f} пикселей")

    import json

    with open('../models/delaunay_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nИтоговые метрики для валидации сохранены в ../models/delaunay_metrics.json")

    print("\nКонец обучения")
