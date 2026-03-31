"""Entry point for the Breast Cancer Detection Flask Application."""
import os
from app import create_app, db, socketio

app = create_app(os.getenv('FLASK_ENV', 'development'))


@app.shell_context_processor
def make_shell_context():
    from app.models.user import User, PatientProfile, DoctorProfile
    from app.models.diagnosis import Diagnosis
    from app.models.consultation import Consultation, ChatMessage
    return {'db': db, 'User': User, 'PatientProfile': PatientProfile,
            'DoctorProfile': DoctorProfile, 'Diagnosis': Diagnosis,
            'Consultation': Consultation, 'ChatMessage': ChatMessage}


@app.cli.command('init-db')
def init_db():
    """Initialize (create) all database tables."""
    with app.app_context():
        db.create_all()
        print('Database tables created.')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=5001)
