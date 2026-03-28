from datetime import datetime
from app import db


class Diagnosis(db.Model):
    __tablename__ = 'diagnoses'

    id              = db.Column(db.Integer, primary_key=True)
    patient_id      = db.Column(db.Integer, db.ForeignKey('patient_profiles.id'), nullable=False)
    image_filename  = db.Column(db.String(256), nullable=False)
    prediction      = db.Column(db.String(32),  nullable=False)   # Benign | Malignant
    is_malignant    = db.Column(db.Boolean,      nullable=False, default=False)
    confidence      = db.Column(db.Float,        nullable=False, default=0.0)
    prob_benign     = db.Column(db.Float,        default=0.0)
    prob_malignant  = db.Column(db.Float,        default=0.0)
    risk_level      = db.Column(db.String(16),   default='Low')
    doctor_notes    = db.Column(db.Text)
    reviewed_by_id  = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    consultation = db.relationship('Consultation', backref='diagnosis',
                                    uselist=False, cascade='all, delete-orphan')
    reviewed_by  = db.relationship('User', foreign_keys=[reviewed_by_id])

    def to_dict(self) -> dict:
        return {
            'id':            self.id,
            'prediction':    self.prediction,
            'is_malignant':  self.is_malignant,
            'confidence':    self.confidence,
            'prob_benign':   self.prob_benign,
            'prob_malignant': self.prob_malignant,
            'risk_level':    self.risk_level,
            'created_at':    self.created_at.strftime('%Y-%m-%d %H:%M'),
        }
