"""
CNN + SVM Hybrid Model for Breast Cancer Detection
===================================================
Architecture:
  • EfficientNetB0  → pre-trained on ImageNet (frozen base)
  • Custom head     → GlobalAveragePooling2D + Dense(512) + Dropout
  • Feature vector  → fed into a fine-tuned SVM (RBF kernel)

BreakHis dataset classes:
  Benign : adenosis (A), fibroadenoma (F), phyllodes_tumor (PT), tubular_adenoma (TA)
  Malignant: ductal_carcinoma (DC), lobular_carcinoma (LC), mucinous_carcinoma (MC),
             papillary_carcinoma (PC)
Binary label : 0 = Benign, 1 = Malignant
"""

import os, joblib, numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build the CNN feature-extractor
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_extractor(img_size: int = 224, trainable_base: bool = False) -> Model:
    """
    Returns a Keras model that outputs a 512-D feature vector.
    The EfficientNetB0 base is frozen by default; call with trainable_base=True
    for fine-tuning after the SVM is trained.
    """
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = trainable_base

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='feature_dense')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    features = layers.Dense(256, activation='relu', name='feature_out')(x)

    model = Model(inputs, features, name='cnn_feature_extractor')
    return model


def build_full_cnn_classifier(img_size: int = 224) -> Model:
    """
    Full CNN classifier (used for initial training / fine-tuning).
    The last Dense(2) head is replaced by SVM in production.
    """
    feat_model = build_feature_extractor(img_size, trainable_base=False)

    inputs = keras.Input(shape=(img_size, img_size, 3))
    features = feat_model(inputs)
    x = layers.Dropout(0.3)(features)
    outputs = layers.Dense(1, activation='sigmoid', name='classifier_head')(x)

    model = Model(inputs, outputs, name='cnn_svm_hybrid')
    return model, feat_model


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SVM classifier wrapper
# ──────────────────────────────────────────────────────────────────────────────
class SVMClassifier:
    """Thin wrapper around sklearn SVC with standard scaling."""

    def __init__(self, kernel='rbf', C=10.0, gamma='scale', probability=True):
        self.scaler = StandardScaler()
        self.svm    = SVC(kernel=kernel, C=C, gamma=gamma,
                          probability=probability, class_weight='balanced',
                          random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svm.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.svm.predict_proba(self.scaler.transform(X))

    def save(self, svm_path: str, scaler_path: str):
        joblib.dump(self.svm,    svm_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, svm_path: str, scaler_path: str) -> 'SVMClassifier':
        obj = cls.__new__(cls)
        obj.svm    = joblib.load(svm_path)
        obj.scaler = joblib.load(scaler_path)
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Hybrid inference class (used by Flask app)
# ──────────────────────────────────────────────────────────────────────────────
class BreastCancerHybridModel:
    """
    End-to-end inference wrapper.
    Loads saved CNN extractor + SVM from disk.
    """

    CLASS_NAMES   = ['Benign', 'Malignant']
    SUBTYPE_MAP   = {
        'benign':    ['Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma'],
        'malignant': ['Ductal Carcinoma', 'Lobular Carcinoma',
                      'Mucinous Carcinoma', 'Papillary Carcinoma']
    }

    def __init__(self, cnn_path: str, svm_path: str, scaler_path: str,
                 img_size: int = 224):
        self.img_size = img_size
        self.cnn      = keras.models.load_model(cnn_path)
        self.svm_clf  = SVMClassifier.load(svm_path, scaler_path)

    # ------------------------------------------------------------------
    def _preprocess(self, img_array: np.ndarray) -> np.ndarray:
        """
        Resize and add batch dim.
        EfficientNetB0 has tf.keras.applications.efficientnet.preprocess_input
        built into the saved model, so pixels must stay in [0, 255].
        Dividing by 255 here causes a train/inference mismatch → always malignant.
        """
        from PIL import Image
        img = Image.fromarray(img_array).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32)       # keep [0, 255] — NOT /255
        return np.expand_dims(arr, axis=0)          # (1, H, W, 3)

    # ------------------------------------------------------------------
    def predict_from_array(self, img_array: np.ndarray) -> dict:
        tensor   = self._preprocess(img_array)
        features = self.cnn.predict(tensor, verbose=0)           # (1, 256)

        # predict_proba columns are ordered by svm.classes_ (may be [0,1] or [1,0])
        raw_proba = self.svm_clf.predict_proba(features)[0]
        classes   = self.svm_clf.svm.classes_                    # e.g. [0, 1]
        proba = np.zeros(2, dtype=np.float32)
        for i, cls in enumerate(classes):
            proba[int(cls)] = raw_proba[i]
        # proba[0] = P(benign), proba[1] = P(malignant) — always correct now

        pred_idx   = int(np.argmax(proba))
        confidence = float(proba[pred_idx]) * 100

        label = self.CLASS_NAMES[pred_idx]
        risk  = _risk_level(proba[1])

        return {
            'prediction':    label,
            'is_malignant':  bool(pred_idx == 1),
            'confidence':    round(confidence, 2),
            'prob_benign':   round(float(proba[0]) * 100, 2),
            'prob_malignant': round(float(proba[1]) * 100, 2),
            'risk_level':    risk,
            'recommendations': _recommendations(pred_idx, proba[1])
        }

    def predict_from_path(self, img_path: str) -> dict:
        import cv2
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.predict_from_array(img)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def _risk_level(prob_malignant: float) -> str:
    if prob_malignant < 0.30:  return 'Low'
    if prob_malignant < 0.60:  return 'Moderate'
    if prob_malignant < 0.80:  return 'High'
    return 'Very High'


def _recommendations(pred_idx: int, prob_mal: float) -> list:
    if pred_idx == 0:
        return [
            'Result suggests benign tissue. Continue routine screening.',
            'Maintain annual mammography schedule.',
            'Consult your doctor for a professional evaluation.'
        ]
    recs = [
        'Immediate consultation with an oncologist is strongly recommended.',
        'Do NOT delay — early treatment significantly improves outcomes.',
        'Provide this report to your doctor during consultation.'
    ]
    if prob_mal > 0.85:
        recs.insert(0, 'HIGH CONFIDENCE malignant indicator detected.')
    return recs


# ──────────────────────────────────────────────────────────────────────────────
# Singleton loader (used by Flask app)
# ──────────────────────────────────────────────────────────────────────────────
_model_instance: BreastCancerHybridModel | None = None

def get_model(config) -> BreastCancerHybridModel | None:
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    # Support both dict-style (Flask app.config) and attribute-style config objects
    def _get(key):
        try:
            return config[key]
        except (TypeError, KeyError):
            return getattr(config, key)
    try:
        _model_instance = BreastCancerHybridModel(
            cnn_path    = _get('CNN_MODEL_FILE'),
            svm_path    = _get('SVM_MODEL_FILE'),
            scaler_path = _get('SCALER_FILE'),
            img_size    = _get('IMG_HEIGHT')
        )
        print('[Model] Hybrid CNN+SVM loaded successfully.')
        return _model_instance
    except Exception as e:
        print(f'[Model] WARNING: Could not load model — {e}')
        print('[Model] Run  python train.py  to train the model first.')
        return None
