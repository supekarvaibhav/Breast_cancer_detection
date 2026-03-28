import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'bc-detect-secret-key-2026-change-in-prod'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(BASE_DIR, 'breast_cancer.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    MODEL_PATH = os.path.join(BASE_DIR, 'app', 'ml', 'saved_models')
    CNN_MODEL_FILE = os.path.join(MODEL_PATH, 'cnn_feature_extractor.keras')
    SVM_MODEL_FILE = os.path.join(MODEL_PATH, 'svm_classifier.joblib')
    SCALER_FILE    = os.path.join(MODEL_PATH, 'feature_scaler.joblib')

    # Image preprocessing settings (BreakHis standard)
    IMG_HEIGHT = 224
    IMG_WIDTH  = 224
    IMG_CHANNELS = 3

    # Training hyper-parameters
    BATCH_SIZE  = 32
    EPOCHS      = 30
    LEARNING_RATE = 1e-4

    # Session lifetime
    PERMANENT_SESSION_LIFETIME = timedelta(hours=12)

    # Mail (optional – fill for email notifications)
    MAIL_SERVER   = 'smtp.gmail.com'
    MAIL_PORT     = 587
    MAIL_USE_TLS  = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
    'default':     DevelopmentConfig
}
