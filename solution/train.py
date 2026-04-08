import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from models import CoordMapper

ROOT_DATA_DIR = '../coord_data'
MODELS_DIR = '../models'
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 1e-4
MLP_MODEL = CoordMapper


def load_split_data(split_name):
    """Загружает координаты из JSON файлов для указанного сплита (train/val)."""
    with open(os.path.join(ROOT_DATA_DIR, 'split.json'), 'r') as f:
        splits = json.load(f)

    session_paths = splits[split_name]

    data_top = {'src': [], 'tgt': []}
    data_bottom = {'src': [], 'tgt': []}

    for session_rel in session_paths:
        session_path = os.path.join(ROOT_DATA_DIR, session_rel)

        # top
        top_json_path = os.path.join(session_path, 'coords_top.json')
        if os.path.exists(top_json_path):
            with open(top_json_path, 'r') as f:
                pairs = json.load(f)
                for pair in pairs:
                    # Собираем пары по совпадению значений number
                    img2_coords = sorted(pair['image2_coordinates'], key=lambda x: x['number'])
                    img1_coords = sorted(pair['image1_coordinates'], key=lambda x: x['number'])

                    for p2, p1 in zip(img2_coords, img1_coords):
                        data_top['src'].append([p2['x'], p2['y']])
                        data_top['tgt'].append([p1['x'], p1['y']])

        # bottom
        bottom_json_path = os.path.join(session_path, 'coords_bottom.json')
        if os.path.exists(bottom_json_path):
            with open(bottom_json_path, 'r') as f:
                pairs = json.load(f)
                for pair in pairs:
                    img2_coords = sorted(pair['image2_coordinates'], key=lambda x: x['number'])
                    img1_coords = sorted(pair['image1_coordinates'], key=lambda x: x['number'])

                    for p2, p1 in zip(img2_coords, img1_coords):
                        data_bottom['src'].append([p2['x'], p2['y']])
                        data_bottom['tgt'].append([p1['x'], p1['y']])

    return {
        'top': (np.array(data_top['src']), np.array(data_top['tgt'])),
        'bottom': (np.array(data_bottom['src']), np.array(data_bottom['tgt']))
    }


def train_model(X, Y, source_name):
    """Обучает модель для одного конкретного источника (top/bottom)."""
    print(f"Обучаем для {source_name} ({len(X)} картинок)")
    # Нормализация/препроцессинг
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    X_norm = scaler_X.fit_transform(X)
    Y_norm = scaler_Y.fit_transform(Y)

    X_t = torch.FloatTensor(X_norm)
    Y_t = torch.FloatTensor(Y_norm)

    # Обучение
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MLP_MODEL()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(loader):.6f}")

    # Сохранение артефактов
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'model_{source_name}.pth'))
    joblib.dump(scaler_X, os.path.join(MODELS_DIR, f'scaler_X_{source_name}.pkl'))
    joblib.dump(scaler_Y, os.path.join(MODELS_DIR, f'scaler_Y_{source_name}.pkl'))
    print(f"Артефакты вохранены в {MODELS_DIR}")

    return model, scaler_X, scaler_Y


def evaluate(model, scaler_X, scaler_Y, X_val, Y_val):
    """Считает MED на валидации"""
    model.eval()
    with torch.no_grad():
        X_val_norm = scaler_X.transform(X_val)
        X_val_t = torch.FloatTensor(X_val_norm)

        pred_norm = model(X_val_t).numpy()
        pred = scaler_Y.inverse_transform(pred_norm)

        # Euclidean Distance
        dist = np.sqrt(np.sum((pred - Y_val) ** 2, axis=1))
        return np.mean(dist)


if __name__ == "__main__":
    data_train = load_split_data('train')
    data_val = load_split_data('val')

    results = {}

    # Обучение и валидация top
    X_train_top, Y_train_top = data_train['top']
    X_val_top, Y_val_top = data_val['top']

    if len(X_train_top) > 0:
        model_top, scX_top, scY_top = train_model(X_train_top, Y_train_top, 'top')
        med_top = evaluate(model_top, scX_top, scY_top, X_val_top, Y_val_top)
        results['top > door2'] = med_top
        print(f"Валидационный MED (top > door2): {med_top:.2f} пикселей")

    # Обучение и валидация bottom
    X_train_bot, Y_train_bot = data_train['bottom']
    X_val_bot, Y_val_bot = data_val['bottom']

    if len(X_train_bot) > 0:
        model_bot, scX_bot, scY_bot = train_model(X_train_bot, Y_train_bot, 'bottom')
        med_bot = evaluate(model_bot, scX_bot, scY_bot, X_val_bot, Y_val_bot)
        results['bottom > door2'] = med_bot
        print(f"Валидационный MED (bottom > door2): {med_bot:.2f} пикселей")

    with open(os.path.join(MODELS_DIR, 'mlp_metrics.json'), 'w') as f:
        print(f"Итоговые метрики для валидации сохранены в {MODELS_DIR} как mlp_metrics.json")
        json.dump(results, f, indent=4)

    print("\nКонец обучения")
