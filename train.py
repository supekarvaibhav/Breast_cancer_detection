"""
train.py  –  Train the CNN + SVM Hybrid Model on BreakHis Dataset
==================================================================
Usage:
    python train.py --dataset /path/to/BreaKHis_v1 --magnification 400X
                    --epochs 30 --batch 32

BreakHis folder structure expected:
    BreaKHis_v1/
      histology_slides/
        breast/
          benign/
            SOB/
              adenosis/        ...  /<mag>/  *.png
              fibroadenoma/
              phyllodes_tumor/
              tubular_adenoma/
          malignant/
            SOB/
              ductal_carcinoma/
              lobular_carcinoma/
              mucinous_carcinoma/
              papillary_carcinoma/
"""

import os, sys, argparse, warnings, random, json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint, TensorBoard)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score, f1_score)
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.ml.model import build_full_cnn_classifier, SVMClassifier

# ─────────────────────────── CONFIG ──────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
SEED       = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

SAVE_DIR = Path(__file__).resolve().parent / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR  = Path(__file__).resolve().parent.parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)


# ─────────────────────────── DATA LOADING ────────────────────────────────────
def collect_image_paths(dataset_root: str, magnification: str = 'ALL') -> tuple:
    """
    Recursively collects (path, label) pairs.
    label: 0 = benign, 1 = malignant
    """
    root   = Path(dataset_root)
    paths, labels = [], []
    mags   = ['40X', '100X', '200X', '400X'] if magnification == 'ALL' \
             else [magnification]

    for label_name, label_int in [('benign', 0), ('malignant', 1)]:
        class_root = root / 'histology_slides' / 'breast' / label_name / 'SOB'
        if not class_root.exists():
            # Fallback: search from root
            class_root = root
        for img_path in class_root.rglob('*.png'):
            if magnification == 'ALL' or any(m in img_path.parts for m in mags):
                paths.append(str(img_path))
                labels.append(label_int)
        for img_path in class_root.rglob('*.jpg'):
            if magnification == 'ALL' or any(m in img_path.parts for m in mags):
                paths.append(str(img_path))
                labels.append(label_int)

    print(f'[Data] Benign: {labels.count(0)}  |  Malignant: {labels.count(1)}  '
          f'|  Total: {len(labels)}')
    return paths, labels


def load_image(path: str, size: int = IMG_SIZE) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def load_dataset(paths: list, labels: list) -> tuple:
    X, y = [], []
    for p, l in tqdm(zip(paths, labels), total=len(paths), desc='Loading images'):
        img = load_image(p)
        if img is not None:
            X.append(img)
            y.append(l)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─────────────────────────── AUGMENTATION ─────────────────────────────────────
def get_augmentor():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )


# ─────────────────────────── PHASE 1 – CNN PRE-TRAINING ──────────────────────
def pretrain_cnn(X_train, y_train, X_val, y_val, epochs: int, batch: int):
    cnn_model, feat_model = build_full_cnn_classifier(IMG_SIZE)

    cnn_model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    cnn_model.summary()

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True,
                      mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(SAVE_DIR / 'best_cnn.keras'), monitor='val_auc',
                        save_best_only=True, mode='max', verbose=1),
        TensorBoard(log_dir=str(LOG_DIR), histogram_freq=1)
    ]

    aug = get_augmentor()
    train_gen = aug.flow(X_train, y_train, batch_size=batch, seed=SEED)

    history = cnn_model.fit(
        train_gen,
        steps_per_epoch=max(1, len(X_train) // batch),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return cnn_model, feat_model, history


# ─────────────────────────── PHASE 2 – FINE-TUNE ─────────────────────────────
def finetune_cnn(cnn_model, X_train, y_train, X_val, y_val, batch: int):
    """Unfreeze top 30 layers of EfficientNetB0 and fine-tune."""
    base = cnn_model.layers[1]          # EfficientNetB0
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    cnn_model.compile(
        optimizer=keras.optimizers.Adam(LR / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True,
                      mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2,
                          min_lr=1e-8, verbose=1),
        ModelCheckpoint(str(SAVE_DIR / 'best_cnn_finetuned.keras'),
                        monitor='val_auc', save_best_only=True,
                        mode='max', verbose=1),
    ]

    aug   = get_augmentor()
    train_gen = aug.flow(X_train, y_train, batch_size=batch, seed=SEED)

    history = cnn_model.fit(
        train_gen,
        steps_per_epoch=max(1, len(X_train) // batch),
        epochs=15,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return cnn_model, history


# ─────────────────────────── PHASE 3 – SVM TRAINING ─────────────────────────
def train_svm(feat_model, X_train, y_train, X_val, y_val, X_test, y_test):
    print('\n[SVM] Extracting CNN features …')
    F_train = feat_model.predict(X_train, batch_size=32, verbose=1)
    F_val   = feat_model.predict(X_val,   batch_size=32, verbose=1)
    F_test  = feat_model.predict(X_test,  batch_size=32, verbose=1)

    F_trval = np.vstack([F_train, F_val])
    y_trval = np.concatenate([y_train, y_val])

    print('[SVM] Training SVM classifier …')
    svm_clf = SVMClassifier(kernel='rbf', C=10.0, gamma='scale')
    svm_clf.fit(F_trval, y_trval)

    # Evaluate
    y_pred  = svm_clf.predict(F_test)
    y_proba = svm_clf.predict_proba(F_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred)

    print(f'\n[SVM] Test Accuracy : {acc*100:.2f}%')
    print(f'[SVM] Test AUC      : {auc*100:.2f}%')
    print(f'[SVM] Test F1-score : {f1*100:.2f}%')
    print('\n[SVM] Classification Report:')
    print(classification_report(y_test, y_pred,
                                 target_names=['Benign','Malignant']))
    return svm_clf, {'accuracy': acc, 'auc': auc, 'f1': f1}


# ─────────────────────────── PLOTTING ────────────────────────────────────────
def plot_metrics(history, tag: str = 'cnn'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'],     label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy'); axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss'); axes[1].legend()

    fig.savefig(str(SAVE_DIR / f'{tag}_training_curves.png'), dpi=150)
    plt.close(fig)
    print(f'[Plot] Saved {tag}_training_curves.png')


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Malignant'],
                yticklabels=['Benign','Malignant'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix – Test Set')
    fig.savefig(str(SAVE_DIR / 'confusion_matrix.png'), dpi=150)
    plt.close(fig)


# ─────────────────────────── MAIN ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train Breast Cancer CNN+SVM')
    parser.add_argument('--dataset', required=True,
                        help='Path to BreaKHis_v1 root folder')
    parser.add_argument('--magnification', default='ALL',
                        choices=['40X','100X','200X','400X','ALL'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch',  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # 1. Load data
    paths, labels = collect_image_paths(args.dataset, args.magnification)
    if len(paths) == 0:
        print('ERROR: No images found. Check dataset path and structure.')
        sys.exit(1)

    X, y = load_dataset(paths, labels)
    print(f'[Data] Dataset shape: {X.shape}, Labels: {y.shape}')

    # 2. Split: 70% train | 15% val | 15% test  (stratified)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, stratify=y_tv, random_state=SEED)
    print(f'[Split] Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}')

    # 3. Pre-train CNN
    print('\n=== PHASE 1: CNN Pre-training ===')
    cnn_model, feat_model, hist1 = pretrain_cnn(
        X_train, y_train, X_val, y_val, args.epochs, args.batch)
    plot_metrics(hist1, 'phase1')

    # 4. Fine-tune CNN
    print('\n=== PHASE 2: CNN Fine-tuning ===')
    cnn_model, hist2 = finetune_cnn(
        cnn_model, X_train, y_train, X_val, y_val, args.batch)
    plot_metrics(hist2, 'phase2')

    # Extract feat_model from fine-tuned cnn_model
    feat_model = keras.Model(
        inputs  = cnn_model.input,
        outputs = cnn_model.get_layer('feature_out').output,
        name    = 'cnn_feature_extractor'
    )

    # 5. Train SVM
    print('\n=== PHASE 3: SVM Training ===')
    svm_clf, metrics = train_svm(
        feat_model, X_train, y_train, X_val, y_val, X_test, y_test)

    # Plot confusion matrix
    F_test  = feat_model.predict(X_test, batch_size=32, verbose=0)
    y_pred  = svm_clf.predict(F_test)
    plot_confusion_matrix(y_test, y_pred)

    # 6. Save models
    feat_model.save(str(SAVE_DIR / 'cnn_feature_extractor.keras'))
    svm_clf.save(
        str(SAVE_DIR / 'svm_classifier.joblib'),
        str(SAVE_DIR / 'feature_scaler.joblib')
    )

    # Save metrics
    with open(str(SAVE_DIR / 'metrics.json'), 'w') as f:
        json.dump({k: round(float(v)*100, 2) for k, v in metrics.items()}, f, indent=2)

    print(f'\n✅  Models saved to  {SAVE_DIR}')
    print(f'   Accuracy : {metrics["accuracy"]*100:.2f}%')
    print(f'   AUC      : {metrics["auc"]*100:.2f}%')
    print(f'   F1-score : {metrics["f1"]*100:.2f}%')


if __name__ == '__main__':
    main()
