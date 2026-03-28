from datetime import datetime
from app import db


class Consultation(db.Model):
    __tablename__ = 'consultations'

    id            = db.Column(db.Integer,  primary_key=True)
    diagnosis_id  = db.Column(db.Integer,  db.ForeignKey('diagnoses.id'), nullable=False)
    patient_id    = db.Column(db.Integer,  db.ForeignKey('patient_profiles.id'), nullable=False)
    doctor_id     = db.Column(db.Integer,  db.ForeignKey('doctor_profiles.id'),  nullable=False)
    status        = db.Column(db.String(32), default='requested')
    # requested | accepted | in_progress | completed | cancelled
    scheduled_at  = db.Column(db.DateTime)
    started_at    = db.Column(db.DateTime)
    ended_at      = db.Column(db.DateTime)
    notes         = db.Column(db.Text)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('PatientProfile', backref='consultations',
                               foreign_keys=[patient_id])

    messages = db.relationship('ChatMessage', backref='consultation',
                                lazy='dynamic', cascade='all, delete-orphan',
                                order_by='ChatMessage.timestamp')

    def duration_minutes(self) -> int | None:
        if self.started_at and self.ended_at:
            return int((self.ended_at - self.started_at).total_seconds() / 60)
        return None


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id              = db.Column(db.Integer, primary_key=True)
    consultation_id = db.Column(db.Integer, db.ForeignKey('consultations.id'), nullable=False)
    sender_id       = db.Column(db.Integer, db.ForeignKey('users.id'),         nullable=False)
    content         = db.Column(db.Text,    nullable=False)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    is_read         = db.Column(db.Boolean,  default=False)

    sender = db.relationship('User', foreign_keys=[sender_id])

    def to_dict(self) -> dict:
        return {
            'id':        self.id,
            'sender':    self.sender.full_name,
            'role':      self.sender.role,
            'content':   self.content,
            'timestamp': self.timestamp.strftime('%H:%M'),
        }
