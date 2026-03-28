from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id            = db.Column(db.Integer,  primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username      = db.Column(db.String(64),  unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(16),  nullable=False)   # 'patient' | 'doctor'
    full_name     = db.Column(db.String(128), nullable=False)
    phone         = db.Column(db.String(20))
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    is_active     = db.Column(db.Boolean,  default=True)
    avatar        = db.Column(db.String(256), default='default_avatar.png')

    # Role-specific relations
    patient_profile = db.relationship('PatientProfile', backref='user',
                                       uselist=False, cascade='all, delete-orphan')
    doctor_profile  = db.relationship('DoctorProfile',  backref='user',
                                       uselist=False, cascade='all, delete-orphan')

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username} [{self.role}]>'


class PatientProfile(db.Model):
    __tablename__ = 'patient_profiles'

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    dob          = db.Column(db.Date)
    gender       = db.Column(db.String(16), default='female')
    address      = db.Column(db.String(256))
    medical_history = db.Column(db.Text)

    # ── Personal Medical History ──────────────────────────────────
    prev_breast_cancer      = db.Column(db.String(3),  default='no')   # yes / no
    prev_breast_lumps       = db.Column(db.String(3),  default='no')
    prev_breast_surgery     = db.Column(db.String(3),  default='no')
    prev_radiation_therapy  = db.Column(db.String(3),  default='no')

    # ── Family History ────────────────────────────────────────────
    family_history_cancer   = db.Column(db.String(3),  default='no')   # yes / no
    family_cancer_members   = db.Column(db.String(128), default='')    # comma-separated: mother,sister,aunt

    # ── Current Symptoms ──────────────────────────────────────────
    symptom_lump            = db.Column(db.Boolean, default=False)
    symptom_size_change     = db.Column(db.Boolean, default=False)
    symptom_nipple_discharge = db.Column(db.Boolean, default=False)
    symptom_skin_changes    = db.Column(db.Boolean, default=False)
    symptom_breast_pain     = db.Column(db.Boolean, default=False)

    # ── Lifestyle Factors ─────────────────────────────────────────
    smoking                 = db.Column(db.String(3),  default='no')
    alcohol                 = db.Column(db.String(3),  default='no')
    physical_activity       = db.Column(db.String(16), default='moderate')  # low / moderate / high

    diagnoses    = db.relationship('Diagnosis', backref='patient',
                                    lazy='dynamic', cascade='all, delete-orphan')


class DoctorProfile(db.Model):
    __tablename__ = 'doctor_profiles'

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    specialization = db.Column(db.String(128), default='Oncology')
    hospital       = db.Column(db.String(128))
    license_number = db.Column(db.String(64))
    years_experience = db.Column(db.Integer, default=0)
    bio            = db.Column(db.Text)
    is_available   = db.Column(db.Boolean, default=True)

    consultations  = db.relationship('Consultation', backref='doctor',
                                      lazy='dynamic', cascade='all, delete-orphan')


@login_manager.user_loader
def load_user(user_id: int) -> User | None:
    return db.session.get(User, int(user_id))
