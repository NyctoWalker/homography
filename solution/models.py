import torch.nn as nn
import pickle
import numpy as np
import cv2

# Интересно, но простой перцептрон имеет наилучшие метрики - остаточные блоки,
# другие функции активации, SIREN и пр. ухудшают результат
class CoordMapper(nn.Module):
    def __init__(self, hidden_size=128):
        super(CoordMapper, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
        )
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        delta = self.output_layer(self.net(x))
        return x + delta


class DelaunayMapper:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.tri = data["triangulation"]
        self.transforms = data["transforms"]
        self.pts_src = data["pts_src"]
        self.pts_dst = data["pts_dst"]

    def predict(self, x, y):
        pt = np.array([x, y])
        simplex_index = self.tri.find_simplex(pt)
        simplex_index = int(simplex_index.item()) if isinstance(simplex_index, np.ndarray) else int(simplex_index)

        if simplex_index != -1:
            M = self.transforms[simplex_index]
            pt_pred = M @ np.array([x, y, 1])
            return float(pt_pred[0]), float(pt_pred[1])

        dists = np.linalg.norm(self.pts_src - pt, axis=1)
        nearest_idx = int(np.argmin(dists))
        for idx, simplex in enumerate(self.tri.simplices):
            if nearest_idx in simplex:
                M = self.transforms[idx]
                pt_pred = M @ np.array([x, y, 1])
                return float(pt_pred[0]), float(pt_pred[1])
        return None, None


class HomographyMapper:
    def __init__(self, model_path):
        self.H = np.load(model_path)

    def predict(self, x, y):
        pt = np.array([[[x, y]]], dtype=np.float32) # shape (1, 1, 2)
        pt_pred = cv2.perspectiveTransform(pt, self.H)
        # (1, 1, 2)
        return float(pt_pred[0][0][0]), float(pt_pred[0][0][1])
