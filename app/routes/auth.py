from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models.user import User, PatientProfile, DoctorProfile

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        role      = request.form.get('role', 'patient')
        email     = request.form.get('email', '').strip().lower()
        username  = request.form.get('username', '').strip()
        full_name = request.form.get('full_name', '').strip()
        phone     = request.form.get('phone', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')

        # Validation
        errors = []
        if not email or '@' not in email:
            errors.append('Valid email is required.')
        if len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')
        if User.query.filter_by(username=username).first():
            errors.append('Username already taken.')

        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('auth/register.html',
                                   form_data=request.form)

        user = User(email=email, username=username,
                    full_name=full_name, phone=phone, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.flush()   # get user.id before commit

        if role == 'patient':
            from datetime import date
            age    = int(request.form.get('age', 0) or 0)
            gender = request.form.get('gender', 'female').strip()
            dob    = date(date.today().year - age, 1, 1) if age > 0 else None

            # Personal Medical History
            prev_breast_cancer     = request.form.get('prev_breast_cancer', 'no')
            prev_breast_lumps      = request.form.get('prev_breast_lumps', 'no')
            prev_breast_surgery    = request.form.get('prev_breast_surgery', 'no')
            prev_radiation_therapy = request.form.get('prev_radiation_therapy', 'no')

            # Family History
            family_history_cancer  = request.form.get('family_history_cancer', 'no')
            family_members_list    = request.form.getlist('family_cancer_members')
            family_cancer_members  = ','.join(family_members_list)

            # Current Symptoms
            symptoms = request.form.getlist('symptoms')

            # Lifestyle Factors
            smoking           = request.form.get('smoking', 'no')
            alcohol           = request.form.get('alcohol', 'no')
            physical_activity = request.form.get('physical_activity', 'moderate')

            # Build legacy medical_history text
            medical_parts = []
            if prev_breast_cancer == 'yes':
                medical_parts.append('Previous breast cancer')
            if prev_breast_lumps == 'yes':
                medical_parts.append('Previous breast lumps/tumors')
            if prev_breast_surgery == 'yes':
                medical_parts.append('History of breast surgery')
            if prev_radiation_therapy == 'yes':
                medical_parts.append('Radiation therapy in chest area')
            if family_history_cancer == 'yes':
                medical_parts.append(f'Family history: {family_cancer_members}')
            if symptoms:
                medical_parts.append(f'Symptoms: {", ".join(symptoms)}')
            medical_history = '; '.join(medical_parts) if medical_parts else ''

            profile = PatientProfile(
                user_id=user.id,
                gender=gender,
                dob=dob,
                medical_history=medical_history,
                # Personal Medical History
                prev_breast_cancer=prev_breast_cancer,
                prev_breast_lumps=prev_breast_lumps,
                prev_breast_surgery=prev_breast_surgery,
                prev_radiation_therapy=prev_radiation_therapy,
                # Family History
                family_history_cancer=family_history_cancer,
                family_cancer_members=family_cancer_members,
                # Current Symptoms
                symptom_lump='lump' in symptoms,
                symptom_size_change='size_change' in symptoms,
                symptom_nipple_discharge='nipple_discharge' in symptoms,
                symptom_skin_changes='skin_changes' in symptoms,
                symptom_breast_pain='breast_pain' in symptoms,
                # Lifestyle
                smoking=smoking,
                alcohol=alcohol,
                physical_activity=physical_activity,
            )
            db.session.add(profile)
        elif role == 'doctor':
            spec  = request.form.get('specialization', 'Oncology').strip()
            hosp  = request.form.get('hospital', '').strip()
            lic   = request.form.get('license_number', '').strip()
            yexp  = int(request.form.get('years_experience', 0) or 0)
            bio   = request.form.get('bio', '').strip()
            profile = DoctorProfile(
                user_id=user.id, specialization=spec,
                hospital=hosp, license_number=lic,
                years_experience=yexp, bio=bio
            )
            db.session.add(profile)

        db.session.commit()
        flash(f'Account created! Welcome, {user.full_name}.', 'success')
        login_user(user)
        return redirect(url_for('patient.dashboard') if role == 'patient'
                        else url_for('doctor.dashboard'))

    return render_template('auth/register.html', form_data={})


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password   = request.form.get('password', '')
        remember   = 'remember' in request.form

        user = (User.query.filter_by(email=identifier).first()
                or User.query.filter_by(username=identifier).first())

        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.full_name}!', 'success')
            next_page = request.args.get('next')
            if not next_page:
                next_page = (url_for('patient.dashboard') if user.role == 'patient'
                             else url_for('doctor.dashboard'))
            return redirect(next_page)
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))
