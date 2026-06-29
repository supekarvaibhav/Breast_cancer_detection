"""Create a test consultation between testpatient and testdoctor for chat testing."""
import os
from dotenv import load_dotenv
load_dotenv()
from app import create_app, db
from app.models.user import User, PatientProfile, DoctorProfile
from app.models.diagnosis import Diagnosis
from app.models.consultation import Consultation

app = create_app(os.getenv('FLASK_ENV', 'development'))

with app.app_context():
    patient_user = User.query.filter_by(username='testpatient').first()
    doctor_user = User.query.filter_by(username='testdoctor').first()

    if not patient_user or not doctor_user:
        print("ERROR: Test users not found. Register them first.")
        exit(1)

    patient_profile = patient_user.patient_profile
    doctor_profile = doctor_user.doctor_profile

    print(f"Patient: {patient_user.full_name} (profile id={patient_profile.id})")
    print(f"Doctor:  {doctor_user.full_name} (profile id={doctor_profile.id})")

    # Check if test diagnosis already exists
    diag = Diagnosis.query.filter_by(patient_id=patient_profile.id).first()
    if not diag:
        diag = Diagnosis(
            patient_id=patient_profile.id,
            image_filename='test_placeholder.png',
            prediction='Malignant',
            is_malignant=True,
            confidence=92.5,
            prob_benign=7.5,
            prob_malignant=92.5,
            risk_level='High',
        )
        db.session.add(diag)
        db.session.commit()
        print(f"Created test diagnosis id={diag.id}")
    else:
        print(f"Using existing diagnosis id={diag.id}")

    # Check if consultation exists
    consult = Consultation.query.filter_by(
        diagnosis_id=diag.id, patient_id=patient_profile.id
    ).first()
    if not consult:
        consult = Consultation(
            diagnosis_id=diag.id,
            patient_id=patient_profile.id,
            doctor_id=doctor_profile.id,
            status='accepted',  # So chat is enabled
        )
        db.session.add(consult)
        db.session.commit()
        print(f"Created test consultation id={consult.id}")
    else:
        print(f"Using existing consultation id={consult.id} (status={consult.status})")

    print(f"\nChat URL: http://127.0.0.1:5001/chat/room/{consult.id}")
