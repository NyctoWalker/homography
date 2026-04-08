import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_single_pair, visualize_warping_results


def homography(img_src, img_dst, pts_src, pts_dst, ransac_threshold=3.0):
    """Находит и применяет гомографию"""
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_threshold)

    if H is None:
        print("Гомография не найдена")
        return None, None, None

    # Применяем гомографию к исходному изображению
    h_dst, w_dst = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w_dst, h_dst))

    # MED
    pts_pred = []
    errors = []
    for i in range(len(pts_src)):
        pt_pred = cv2.perspectiveTransform(pts_src[i].reshape(-1, 1, 2), H)[0][0]
        pts_pred.append(pt_pred)
        error = np.linalg.norm(pt_pred - pts_dst[i])
        errors.append(error)

    pts_pred = np.array(pts_pred)
    med = np.mean(errors)
    print(f"MED (Mean Euclidean Distance): {med:.2f} px")

    return warped, H, med, pts_pred


def visualize_homography(img_src, img_dst, img_warped, pts_src, pts_dst, H=None, med=None):
    """Визуализирует результаты гомографии"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # top/bottom с точками
    axes[0, 0].imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
    axes[0, 0].scatter(pts_src[:, 0], pts_src[:, 1], c='red', s=50, marker='o')
    axes[0, 0].axis('off')

    # door2 точками
    axes[0, 1].imshow(cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB))
    axes[0, 1].scatter(pts_dst[:, 0], pts_dst[:, 1], c='green', s=50, marker='o')
    axes[0, 1].axis('off')

    # Искажённое изображение
    axes[1, 0].imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
    axes[1, 0].axis('off')

    # Сравнение точек на целевом изображении
    axes[1, 1].imshow(cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB))

    # Предсказанные точки из гомографии
    if H is not None:
        pts_pred = []
        for pt in pts_src:
            pt_pred = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), H)[0][0]
            pts_pred.append(pt_pred)
        pts_pred = np.array(pts_pred)

        axes[1, 1].scatter(pts_dst[:, 0], pts_dst[:, 1], c='lime', s=80, marker='o', label='Ground Truth')
        axes[1, 1].scatter(pts_pred[:, 0], pts_pred[:, 1], c='red', s=50, marker='x', label='Predicted')

        # Линии ошибок
        for i in range(len(pts_dst)):
            axes[1, 1].plot([pts_dst[i, 0], pts_pred[i, 0]],
                            [pts_dst[i, 1], pts_pred[i, 1]],
                            'yellow', linewidth=1)

        if med is not None:
            axes[1, 1].text(10, 30, f"MED: {med:.2f} px", fontsize=12, color='white',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black'))

    axes[1, 1].set_title("Сравнение GT и предсказаний")
    axes[1, 1].legend()
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ROOT_DATA_DIR = '../coord_data'

    # Ищем первую доступную сессию
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
        # Применяем гомографию
        img_warped, H, med, pts_pred = homography(img_src, img_dst, pts_src, pts_dst, ransac_threshold=1.0)

        if img_warped is not None:
            visualize_warping_results(img_src, img_dst, img_warped, pts_src, pts_dst,
                                      method_name="Гомография", pts_pred=pts_pred, med=med)
        else:
            print("Не удалось найти гомографию")
    else:
        print("Не найдена пара картинок")
