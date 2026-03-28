"""
create_demo_model.py
=====================
Creates a lightweight demo CNN+SVM model so the Flask website
works immediately without the full BreakHis training run.

Usage:
    python create_demo_model.py

This creates small placeholder model files in app/ml/saved_models/
that demonstrate the prediction pipeline. Replace them with
properly trained models by running:
    python train.py --dataset /path/to/BreaKHis_v1
"""

import os, sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAVE_DIR = Path(__file__).resolve().parent / 'app' / 'ml' / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def create_demo():
    print('[Demo] Building lightweight demo CNN feature extractor...')

    # ── Build a tiny EfficientNetB0-based extractor ───────────────────
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0

    base = EfficientNetB0(include_top=False, weights='imagenet',
                          input_shape=(224, 224, 3))
    base.trainable = False

    inp  = keras.Input(shape=(224, 224, 3))
    x    = base(inp, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(512, activation='relu', name='feature_dense')(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Dropout(0.4)(x)
    feat = layers.Dense(256, activation='relu', name='feature_out')(x)

    cnn = keras.Model(inp, feat, name='cnn_feature_extractor')
    cnn_path = str(SAVE_DIR / 'cnn_feature_extractor.keras')
    cnn.save(cnn_path)
    print(f'[Demo] CNN saved → {cnn_path}')

    # ── Create and save a quick SVM on random data ──────────────────
    print('[Demo] Training demo SVM on synthetic data...')
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import joblib

    rng   = np.random.default_rng(42)
    X_syn = rng.standard_normal((200, 256)).astype(np.float32)
    y_syn = rng.integers(0, 2, size=200)

    scaler = StandardScaler().fit(X_syn)
    X_s    = scaler.transform(X_syn)
    svm    = SVC(kernel='rbf', C=10, probability=True, random_state=42)
    svm.fit(X_s, y_syn)

    svm_path    = str(SAVE_DIR / 'svm_classifier.joblib')
    scaler_path = str(SAVE_DIR / 'feature_scaler.joblib')
    joblib.dump(svm,    svm_path)
    joblib.dump(scaler, scaler_path)
    print(f'[Demo] SVM    saved → {svm_path}')
    print(f'[Demo] Scaler saved → {scaler_path}')

    # ── Metrics placeholder ─────────────────────────────────────────
    metrics = {'accuracy': 0.0, 'auc': 0.0, 'f1': 0.0,
               'note': 'Demo model – replace with trained model'}
    with open(str(SAVE_DIR / 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\n✅  Demo model ready!')
    print('   Run  python train.py --dataset /path/to/BreaKHis_v1  for the real model.')
    print('   Then start the server:  python run.py')


if __name__ == '__main__':
    create_demo()
