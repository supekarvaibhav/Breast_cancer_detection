from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_socketio import SocketIO
from flask_migrate import Migrate
from config import config

db       = SQLAlchemy()
login_manager = LoginManager()
socketio = SocketIO()
migrate  = Migrate()


def create_app(config_name: str = 'default') -> Flask:
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Extensions
    db.init_app(app)
    login_manager.init_app(app)
    socketio.init_app(app, cors_allowed_origins='*', async_mode='eventlet')
    migrate.init_app(app, db)

    login_manager.login_view   = 'auth.login'
    login_manager.login_message_category = 'info'

    # Blueprints
    from .routes.auth    import auth_bp
    from .routes.patient import patient_bp
    from .routes.doctor  import doctor_bp
    from .routes.main    import main_bp
    from .routes.chat    import chat_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp,  url_prefix='/patient')
    app.register_blueprint(doctor_bp,   url_prefix='/doctor')
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp,     url_prefix='/chat')

    # Ensure upload folder exists
    import os
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_PATH'],    exist_ok=True)

    # Pre-load model
    with app.app_context():
        from .ml.model import get_model
        app.ml_model = get_model(app.config)

    return app
