import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_single_pair(session_dir, json_file, pair_index=0):
    """Загружает одну пару изображений и их точки разметки."""
    json_path = os.path.join(session_dir, json_file)

    if not os.path.exists(json_path):
        print(f"Файл не найден: {json_path}")
        return None, None, None, None

    with open(json_path, 'r') as f:
        data = json.load(f)

    if pair_index >= len(data):
        return None, None, None, None

    item = data[pair_index]
    subfolder = 'top' if 'top' in json_file else 'bottom'

    # Извлекаем пути к файлам, отсекая префиксы сессий
    fname1 = os.path.basename(item.get('file1_path', ''))
    fname2 = os.path.basename(item.get('file2_path', ''))

    img1_path = os.path.join(session_dir, 'door2', fname1)
    img2_path = os.path.join(session_dir, subfolder, fname2)

    img1 = cv2.imread(img1_path)  # door2
    img2 = cv2.imread(img2_path)  # top/bottom

    if img1 is None or img2 is None:
        print(f"Пара картинок не найдена:\n- {img1_path}\n- {img2_path}")
        return None, None, None, None

    coord1 = item.get('image1_coordinates', [])  # door2
    coord2 = item.get('image2_coordinates', [])  # top/bottom

    # Есть данные где для одной пары картинок точек больше, берём срез по минимуму чтобы не терять данные
    if len(coord1) != len(coord2):
        _min = min(len(coord1), len(coord2))
        coord1 = coord1[:_min]
        coord2 = coord2[:_min]

    # Нужно минимум 4 опорных точки, остальные сценарии игнорируем
    if len(coord1) < 4:
        print(f"Слишком мало пар точек ({len(coord1)})")
        return None, None, None, None

    pts_dst = np.array([[p['x'], p['y']] for p in coord1], dtype=np.float64)  # door2
    pts_src = np.array([[p['x'], p['y']] for p in coord2], dtype=np.float64)  # top/bottom

    print(f"{len(pts_src)} пар точек")
    return img1, img2, pts_src, pts_dst


def visualize_warping_results(img_src, img_dst, img_warped, pts_src, pts_dst,
                              method_name="Warping", pts_pred=None, med=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # bottom/top с точками
    axes[0, 0].imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
    axes[0, 0].scatter(pts_src[:, 0], pts_src[:, 1], c='red', s=50, marker='o')
    axes[0, 0].set_title(f"Top/bottom")
    axes[0, 0].axis('off')

    # door2 с точками (ground truth)
    axes[0, 1].imshow(cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB))
    axes[0, 1].scatter(pts_dst[:, 0], pts_dst[:, 1], c='lime', s=50, marker='o')
    axes[0, 1].set_title(f"door2")
    axes[0, 1].axis('off')

    # Искажённое bottom/top
    axes[1, 0].imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Искажение ({method_name})")
    axes[1, 0].axis('off')

    # Сравнение
    if pts_pred is not None:
        axes[1, 1].imshow(cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB))
        axes[1, 1].scatter(pts_dst[:, 0], pts_dst[:, 1], c='lime', s=80, marker='o', label='Ground Truth')
        axes[1, 1].scatter(pts_pred[:, 0], pts_pred[:, 1], c='red', s=50, marker='x', label='Predicted')

        for i in range(len(pts_dst)):
            axes[1, 1].plot([pts_dst[i, 0], pts_pred[i, 0]], [pts_dst[i, 1], pts_pred[i, 1]],'yellow', linewidth=1)

        if med is not None:
            axes[1, 1].text(10, 30, f"MED: {med:.2f} px", fontsize=12, color='white',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black'))
        axes[1, 1].legend(loc='upper right')
    else:
        blended = cv2.addWeighted(img_dst, 0.5, img_warped, 0.5, 0)
        axes[1, 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[1, 1].scatter(pts_dst[:, 0], pts_dst[:, 1], c='yellow', s=50, marker='o', alpha=0.7)

    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
