import os
import json
import cv2
import numpy as np

ROOT_DATA_DIR = '../coord_data'
MODELS_DIR = '../models'


def load_all_points(split_name):
    """Собирает все точки из train сплита для расчета гомографии"""
    with open(os.path.join(ROOT_DATA_DIR, 'split.json'), 'r') as f:
        splits = json.load(f)

    session_paths = splits.get(split_name, [])

    data_top = {'src': [], 'tgt': []}
    data_bottom = {'src': [], 'tgt': []}

    for session_rel in session_paths:
        session_path = os.path.join(ROOT_DATA_DIR, session_rel)

        # Обрабатываем top и bottom
        for cam_type, data_dict in [('top', data_top), ('bottom', data_bottom)]:
            json_path = os.path.join(session_path, f'coords_{cam_type}.json')
            if not os.path.exists(json_path): continue

            with open(json_path, 'r') as f:
                pairs = json.load(f)

            for pair in pairs:
                src_pts = sorted(pair['image2_coordinates'], key=lambda x: x['number'])
                tgt_pts = sorted(pair['image1_coordinates'], key=lambda x: x['number'])

                for s, t in zip(src_pts, tgt_pts):
                    # Проверка совпадения ID для надежности
                    if s['number'] == t['number']:
                        data_dict['src'].append([s['x'], s['y']])
                        data_dict['tgt'].append([t['x'], t['y']])

    for key in data_top: data_top[key] = np.array(data_top[key], dtype=np.float32)
    for key in data_bottom: data_bottom[key] = np.array(data_bottom[key], dtype=np.float32)

    return data_top, data_bottom


def train_homography(source_points, target_points, source_name):
    """Вычисляет глобальную матрицу гомографии"""
    if len(source_points) < 4:
        print(f"Недостаточно точек для {source_name}: {len(source_points)}")
        return None

    print(f"Вычисляем гомографию на {source_name} на {len(source_points)} точках")

    H, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

    if H is None:
        print(f"Не удалось найти гомографию для {source_name}")
        return None

    # Считаем MED на тренировочных данных
    pts_src_t = source_points.reshape(-1, 1, 2)
    pts_pred = cv2.perspectiveTransform(pts_src_t, H).reshape(-1, 2)

    errors = np.linalg.norm(pts_pred - target_points, axis=1)
    med = np.mean(errors)

    inliers = np.sum(mask)

    print(f"{source_name}: MED = {med:.2f} px, RANSAC inliers: {inliers}/{len(source_points)}")

    return H


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    data_top, data_bottom = load_all_points('train')
    results = {}

    # top
    H_top = train_homography(data_top['src'], data_top['tgt'], 'top')
    if H_top is not None:
        np.save(os.path.join(MODELS_DIR, 'homography_top.npy'), H_top)

    # bottom
    H_bot = train_homography(data_bottom['src'], data_bottom['tgt'], 'bottom')
    if H_bot is not None:
        np.save(os.path.join(MODELS_DIR, 'homography_bottom.npy'), H_bot)

    print("\nГомографии сохранены в папку models/")
