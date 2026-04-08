import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

from predict import predict


def load_val_data(root_dir):
    """Загружает данные из val"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_root_dir = os.path.join(script_dir, root_dir)

    val_data = []
    val_dir = os.path.join(abs_root_dir, 'val')

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Папка {val_dir} не найдена")

    for session in os.listdir(val_dir):
        sess_path = os.path.join(val_dir, session)
        if not os.path.isdir(sess_path): continue

        for cam_type in ['top', 'bottom']:
            json_path = os.path.join(sess_path, f'coords_{cam_type}.json')
            if not os.path.exists(json_path): continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            for item in data:
                # Отсекаем префикс сессии
                fname1 = os.path.basename(item.get('file1_path', ''))
                img_path = os.path.join(sess_path, 'door2', fname1)

                c1 = item.get('image1_coordinates', [])  # GT (door2)
                c2 = item.get('image2_coordinates', [])  # Source (top/bot)

                if len(c1) != len(c2): continue

                val_data.append({
                    'source': cam_type,
                    'img_path': img_path,
                    'gt': c1,
                    'src': c2
                })
    return val_data


def evaluate_and_visualize(args):
    print(f"Загрузка валидационных данных из {args.data_dir}...")
    val_data = load_val_data(args.data_dir)
    print(f"Найдено пар: {len(val_data)}")

    errors = {"top": [], "bottom": []}
    vis_collection = []

    print(f"Вычисление предиктов (метод: {args.method})...")
    for item in val_data:
        source = item['source']
        item_errors = []
        pts_gt = []
        pts_pred = []

        for p_gt, p_src in zip(item['gt'], item['src']):
            x_pred, y_pred = predict(p_src['x'], p_src['y'], source, method=args.method)

            if x_pred is not None:
                err = np.sqrt((x_pred - p_gt['x']) ** 2 + (y_pred - p_gt['y']) ** 2)
                item_errors.append(err)
                errors[source].append(err)
                pts_gt.append([p_gt['x'], p_gt['y']])
                pts_pred.append([x_pred, y_pred])

        # Собираем данные для визуализации, если не превышен лимит визуализации
        if args.visualize and len(vis_collection) < args.visualization_limit and len(pts_gt) > 0:
            vis_collection.append({
                'source': source,
                'img_path': item['img_path'],
                'gt': np.array(pts_gt),
                'pred': np.array(pts_pred),
                'mean_err': np.mean(item_errors)
            })

    # Метрики
    med_top = np.mean(errors["top"]) if errors["top"] else 0
    med_bottom = np.mean(errors["bottom"]) if errors["bottom"] else 0

    output_str = (
        f"method: {args.method}\n\n"
        f"MED top > door2: {med_top:.2f} px\n"
        f"MED bottom > door2: {med_bottom:.2f} px\n"
        f"Average MED: {(med_top + med_bottom) / 2:.2f} px\n"
    )

    with open(args.output_file, 'w') as f:
        f.write(output_str)
    print(f"\nМетрики сохранены в {args.output_file}")
    print(output_str)

    # Визуализация
    if args.visualize and vis_collection:
        fig, axes = plt.subplots(1, len(vis_collection), figsize=(8 * len(vis_collection), 8))
        if len(vis_collection) == 1: axes = [axes]

        for i, vis_data in enumerate(vis_collection):
            ax = axes[i]

            img = cv2.imread(vis_data['img_path'])
            if img is not None:
                h, w = img.shape[:2]
                scale = 800 / max(h, w)
                img_res = cv2.resize(img, (int(w * scale), int(h * scale)))
                ax.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))

                gt_vis = vis_data['gt'] * scale
                pred_vis = vis_data['pred'] * scale
            else:
                ax.set_facecolor('black')
                gt_vis = vis_data['gt']
                pred_vis = vis_data['pred']

            for g, p in zip(gt_vis, pred_vis):
                ax.plot([g[0], p[0]], [g[1], p[1]], color='yellow', linewidth=1)

            ax.scatter(gt_vis[:, 0], gt_vis[:, 1], c='lime', s=30, marker='o', label='Ground Truth')
            ax.scatter(pred_vis[:, 0], pred_vis[:, 1], c='red', s=30, marker='x', label='Predicted')

            ax.set_title(f"{vis_data['source'].upper()} > door2\nMean Err: {vis_data['mean_err']:.2f} px", color='white')
            ax.legend(loc='upper right')
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation скрипт для маппера координат")
    parser.add_argument('--data_dir', type=str, default='../coord_data',
                        help="Путь к корневой папке с данными")
    parser.add_argument('--method', type=str, default='mlp', choices=['delaunay', 'mlp', 'homography'],
                        help="Используемый метод")
    parser.add_argument('--output_file', type=str, default='metrics.txt', help="Файл для сохранения метрик")

    parser.add_argument('--visualize', action='store_true', help="Включает визуализацию")
    parser.add_argument('--visualization_limit', type=int, default=2,
                        help="Максимальное количество отображаемых картинок")

    args = parser.parse_args()

    evaluate_and_visualize(args)
    # python solution/eval.py --visualize --visualization_limit 4 --method "mlp"
