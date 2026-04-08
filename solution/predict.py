import os
import torch
import joblib
from models import CoordMapper, DelaunayMapper, HomographyMapper
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLP_MODEL = CoordMapper


class Predictor:
    _instance = None

    def __new__(cls, method='delaunay', models_dir='../models'):
        if cls._instance is None:
            cls._instance = super(Predictor, cls).__new__(cls)
            cls._instance._init(method, models_dir)
        return cls._instance

    def _init(self, method, models_dir):
        self.method = method.lower()
        self.mappers = {}
        self.scalers = {}
        abs_models_dir = os.path.join(BASE_DIR, models_dir)

        if self.method == 'delaunay':
            self.mappers['top'] = DelaunayMapper(os.path.join(abs_models_dir, 'delaunay_top.pkl'))
            self.mappers['bottom'] = DelaunayMapper(os.path.join(abs_models_dir, 'delaunay_bottom.pkl'))

        elif self.method == 'mlp':
            self.device = torch.device('cpu')
            for source in ['top', 'bottom']:
                model_path = os.path.join(abs_models_dir, f'model_{source}.pth')
                if os.path.exists(model_path):
                    model = MLP_MODEL()
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    self.mappers[source] = model
                    self.scalers[source] = {
                        'X': joblib.load(os.path.join(abs_models_dir, f'scaler_X_{source}.pkl')),
                        'Y': joblib.load(os.path.join(abs_models_dir, f'scaler_Y_{source}.pkl'))
                    }

        elif self.method == 'homography':
            for source in ['top', 'bottom']:
                h_path = os.path.join(abs_models_dir, f'homography_{source}.npy')
                if os.path.exists(h_path):
                    self.mappers[source] = HomographyMapper(h_path)
                else:
                    print(f"Матрицы гомографии в {h_path} не найдены")

    def predict(self, x, y, source):
        if source not in self.mappers:
            raise ValueError(f"Unknown source: {source}")

        if self.method == 'delaunay':
            return self.mappers[source].predict(x, y)
        elif self.method == 'mlp':
            model = self.mappers[source]
            point_norm = self.scalers[source]['X'].transform([[x, y]])
            with torch.no_grad():
                pred_norm = model(torch.FloatTensor(point_norm)).numpy()
            pred = self.scalers[source]['Y'].inverse_transform(pred_norm)
            return float(pred[0][0]), float(pred[0][1])
        elif self.method == 'homography':
            return self.mappers[source].predict(x, y)


def predict(x, y, source, method='mlp'):
    """
    Parameters:
        x: координата x на кадре верхней/нижней камеры.
        y: координата y на кадре верхней/нижней камеры.
        source: "top" или "bottom".
        method: "mlp" (по умолчанию), "homography" или "delaunay" для выбора метода.
    Returns:
        (x', y') - предсказанная пиксельная координата на кадре door2
    """
    predictor = Predictor(method=method)
    return predictor.predict(x, y, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Получение предсказанной координаты на door2 по координате с top/bottom камеры")

    parser.add_argument('-x', type=float, required=True, help="Координата x на кадре источника")
    parser.add_argument('-y', type=float, required=True, help="Координата y на кадре источника")
    parser.add_argument('-s', '--source', type=str, required=True, choices=['top', 'bottom'],
                        help="Источник: 'top' или 'bottom'")

    # Опционально
    parser.add_argument('-m', '--method', type=str, default='mlp',
                        choices=['delaunay', 'mlp', 'homography'], help="Метод маппинга (по умолчанию: mlp)")
    parser.add_argument('--print_result', action=argparse.BooleanOptionalAction, default=True,
                        help="Выводить ли результат на экран (по умолчанию: True, флаг --no-print_result чтобы отключить)")

    args = parser.parse_args()

    # Вызов функции
    try:
        x_pred, y_pred = predict(args.x, args.y, args.source, method=args.method)

        if args.print_result:
            print(f"{round(x_pred, 2)}, {round(y_pred, 2)}")

    except Exception as e:
        print(f"Ошибка: {e}")
        exit(1)
