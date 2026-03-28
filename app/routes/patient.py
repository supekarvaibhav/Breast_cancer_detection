import os, uuid
from flask import (Blueprint, render_template, redirect, url_for, flash,
                   request, current_app, jsonify)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from app import db
from app.models.user import PatientProfile, DoctorProfile
from app.models.diagnosis import Diagnosis
from app.models.consultation import Consultation

patient_bp = Blueprint('patient', __name__)


def _allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower()
            in current_app.config['ALLOWED_EXTENSIONS'])


def _require_patient(func):
    """Decorator: ensure logged-in user is a patient."""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'patient':
            flash('Access denied.', 'danger')
            return redirect(url_for('main.index'))
        return func(*args, **kwargs)
    return login_required(wrapper)


@patient_bp.route('/dashboard')
@_require_patient
def dashboard():
    profile = current_user.patient_profile
    recent  = (profile.diagnoses.order_by(Diagnosis.created_at.desc())
               .limit(5).all() if profile else [])
    stats   = {
        'total':    profile.diagnoses.count()            if profile else 0,
        'malignant':profile.diagnoses.filter_by(is_malignant=True).count() if profile else 0,
        'benign':   profile.diagnoses.filter_by(is_malignant=False).count() if profile else 0,
    }
    return render_template('patient/dashboard.html',
                           recent_diagnoses=recent, stats=stats)


@patient_bp.route('/upload', methods=['GET', 'POST'])
@_require_patient
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if not _allowed_file(file.filename):
            flash('Invalid file type. Allowed: PNG, JPG, JPEG, TIF, BMP', 'danger')
            return redirect(request.url)

        # Save with unique name
        ext      = secure_filename(file.filename).rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Run AI inference
        model = current_app.ml_model
        if model is None:
            flash('AI model is not loaded. Please contact administrator.', 'warning')
            return redirect(url_for('patient.dashboard'))

        try:
            result = model.predict_from_path(save_path)
        except Exception as e:
            flash(f'Analysis failed: {e}', 'danger')
            return redirect(request.url)

        # Save diagnosis to DB
        profile   = current_user.patient_profile
        diagnosis = Diagnosis(
            patient_id     = profile.id,
            image_filename = filename,
            prediction     = result['prediction'],
            is_malignant   = result['is_malignant'],
            confidence     = result['confidence'],
            prob_benign    = result['prob_benign'],
            prob_malignant = result['prob_malignant'],
            risk_level     = result['risk_level'],
        )
        db.session.add(diagnosis)
        db.session.commit()

        flash('Analysis complete!', 'success')
        return redirect(url_for('patient.result', diagnosis_id=diagnosis.id))

    return render_template('patient/upload.html')


@patient_bp.route('/result/<int:diagnosis_id>')
@_require_patient
def result(diagnosis_id: int):
    diag = Diagnosis.query.get_or_404(diagnosis_id)
    if diag.patient.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('patient.dashboard'))

    # Fetch available doctors for consultation
    available_doctors = []
    if diag.is_malignant:
        available_doctors = (DoctorProfile.query
                             .filter_by(is_available=True)
                             .join(DoctorProfile.user).all())

    recommendations = {
        'Low':       'Continue routine screening. No immediate action needed.',
        'Moderate':  'Schedule a follow-up with your doctor.',
        'High':      'Consult an oncologist as soon as possible.',
        'Very High': 'URGENT: Seek immediate medical consultation.'
    }.get(diag.risk_level, '')

    return render_template('patient/result.html',
                           diag=diag,
                           available_doctors=available_doctors,
                           recommendation=recommendations)


@patient_bp.route('/request-consultation/<int:diagnosis_id>/<int:doctor_id>',
                  methods=['POST'])
@_require_patient
def request_consultation(diagnosis_id: int, doctor_id: int):
    diag    = Diagnosis.query.get_or_404(diagnosis_id)
    doctor  = DoctorProfile.query.get_or_404(doctor_id)
    profile = current_user.patient_profile

    if diag.patient_id != profile.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('patient.dashboard'))

    # Check if consultation already exists
    existing = (Consultation.query
                .filter_by(diagnosis_id=diag.id, patient_id=profile.id)
                .first())
    if existing:
        flash('Consultation already requested.', 'info')
        return redirect(url_for('chat.room', consultation_id=existing.id))

    consult = Consultation(
        diagnosis_id = diag.id,
        patient_id   = profile.id,
        doctor_id    = doctor.id,
        status       = 'requested'
    )
    db.session.add(consult)
    db.session.commit()
    flash(f'Consultation requested with Dr. {doctor.user.full_name}!', 'success')
    return redirect(url_for('chat.room', consultation_id=consult.id))


@patient_bp.route('/history')
@_require_patient
def history():
    profile = current_user.patient_profile
    page    = request.args.get('page', 1, type=int)
    diags   = (profile.diagnoses.order_by(Diagnosis.created_at.desc())
               .paginate(page=page, per_page=10, error_out=False)
               if profile else None)
    return render_template('patient/history.html', pagination=diags)
