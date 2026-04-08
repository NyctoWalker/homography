import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from utils import load_single_pair, visualize_warping_results


def apply_tps_warping(img_src, img_dst, pts_src, pts_dst, smoothing=5.0):
    """Применяет Thin Plate Spline для искажения img_src в соответствии с img_dst(door2)"""
    h_dst, w_dst = img_dst.shape[:2]

    interp_x = RBFInterpolator(pts_dst, pts_src[:, 0], kernel='thin_plate_spline', smoothing=smoothing)
    interp_y = RBFInterpolator(pts_dst, pts_src[:, 1], kernel='thin_plate_spline', smoothing=smoothing)

    y_grid, x_grid = np.mgrid[0:h_dst, 0:w_dst]
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel())).astype(np.float64)

    print("Расчёт TPS-сетки (занимает какое-то время)")

    # Для каждого пикселя door2 находим его координаты на top/bottom
    map_x = interp_x(grid_points).reshape(h_dst, w_dst).astype(np.float32)
    map_y = interp_y(grid_points).reshape(h_dst, w_dst).astype(np.float32)

    # Искажаем с помощью обратного отображения
    warped = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return warped


if __name__ == "__main__":
    ROOT_DATA_DIR = '../coord_data'

    session_to_use = None
    for split in ['train', 'val']:
        split_dir = os.path.join(ROOT_DATA_DIR, split)
        if os.path.exists(split_dir):
            for item in os.listdir(split_dir):
                if os.path.isdir(os.path.join(split_dir, item)):
                    session_to_use = os.path.join(split_dir, item)
                    break
            if session_to_use:
                break

    if not session_to_use:
        print("Папок сессий не найдено, нужно проверить путь к файлам")
        exit()

    print(f"Сессия {session_to_use}")

    # top для примера работы, т.к. хорошо видно искажения коробок
    json_file = 'coords_top.json'
    img_dst, img_src, pts_src, pts_dst = load_single_pair(session_to_use, json_file, pair_index=0)

    if img_src is not None and img_dst is not None:
        # Если точки разметки имеют погрешности (шум), можно поставить smoothing=1.0 или 10.0
        img_warped = apply_tps_warping(img_src, img_dst, pts_src, pts_dst, smoothing=10.0)

        visualize_warping_results(img_src, img_dst, img_warped, pts_src, pts_dst, method_name="Thin Plate Spline")
    else:
        print("Не найдена пара картинок")
