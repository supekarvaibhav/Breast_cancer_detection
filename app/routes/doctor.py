from flask import (Blueprint, render_template, redirect, url_for, flash,
                   request, current_app, jsonify)
from flask_login import login_required, current_user
from datetime import datetime

from app import db
from app.models.user import DoctorProfile
from app.models.diagnosis import Diagnosis
from app.models.consultation import Consultation

doctor_bp = Blueprint('doctor', __name__)


def _require_doctor(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'doctor':
            flash('Doctor access required.', 'danger')
            return redirect(url_for('main.index'))
        return func(*args, **kwargs)
    return login_required(wrapper)


@doctor_bp.route('/dashboard')
@_require_doctor
def dashboard():
    profile = current_user.doctor_profile
    pending = (Consultation.query
               .filter_by(doctor_id=profile.id, status='requested')
               .order_by(Consultation.created_at.desc()).all()
               if profile else [])
    active  = (Consultation.query
               .filter(Consultation.doctor_id == profile.id,
                       Consultation.status.in_(['accepted', 'in_progress']))
               .order_by(Consultation.created_at.desc()).all()
               if profile else [])
    stats = {
        'total':     profile.consultations.count()              if profile else 0,
        'pending':   len(pending),
        'completed': profile.consultations.filter_by(status='completed').count() if profile else 0,
    }
    return render_template('doctor/dashboard.html',
                           pending=pending, active=active, stats=stats)


@doctor_bp.route('/consultation/<int:consult_id>/accept', methods=['POST'])
@_require_doctor
def accept_consultation(consult_id: int):
    consult = Consultation.query.get_or_404(consult_id)
    profile = current_user.doctor_profile
    if consult.doctor_id != profile.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('doctor.dashboard'))
    consult.status     = 'accepted'
    consult.started_at = datetime.utcnow()
    db.session.commit()
    flash('Consultation accepted.', 'success')
    return redirect(url_for('chat.room', consultation_id=consult.id))


@doctor_bp.route('/consultation/<int:consult_id>/complete', methods=['POST'])
@_require_doctor
def complete_consultation(consult_id: int):
    consult = Consultation.query.get_or_404(consult_id)
    profile = current_user.doctor_profile
    if consult.doctor_id != profile.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('doctor.dashboard'))
    consult.status   = 'completed'
    consult.ended_at = datetime.utcnow()
    consult.notes    = request.form.get('doctor_notes', '')

    # Also update diagnosis with doctor notes
    if consult.diagnosis:
        consult.diagnosis.doctor_notes   = consult.notes
        consult.diagnosis.reviewed_by_id = current_user.id

    db.session.commit()
    flash('Consultation completed and notes saved.', 'success')
    return redirect(url_for('doctor.dashboard'))


@doctor_bp.route('/patients')
@_require_doctor
def patients():
    profile = current_user.doctor_profile
    consults = (profile.consultations
                .filter(Consultation.status == 'completed')
                .order_by(Consultation.ended_at.desc()).all()
                if profile else [])
    return render_template('doctor/patients.html', consultations=consults)


@doctor_bp.route('/toggle-availability', methods=['POST'])
@_require_doctor
def toggle_availability():
    profile = current_user.doctor_profile
    profile.is_available = not profile.is_available
    db.session.commit()
    status = 'available' if profile.is_available else 'unavailable'
    flash(f'You are now {status} for new consultations.', 'info')
    return redirect(url_for('doctor.dashboard'))


@doctor_bp.route('/diagnosis/<int:diag_id>')
@_require_doctor
def view_diagnosis(diag_id: int):
    diag = Diagnosis.query.get_or_404(diag_id)
    return render_template('doctor/view_diagnosis.html', diag=diag)
