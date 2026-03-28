from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from flask_socketio import emit, join_room, leave_room
from datetime import datetime

from app import db, socketio
from app.models.consultation import Consultation, ChatMessage

chat_bp = Blueprint('chat', __name__)


def _can_access_consultation(consult: Consultation) -> bool:
    if current_user.role == 'patient':
        return consult.patient.user_id == current_user.id
    elif current_user.role == 'doctor':
        return consult.doctor.user_id == current_user.id
    return False


@chat_bp.route('/room/<int:consultation_id>')
@login_required
def room(consultation_id: int):
    consult = Consultation.query.get_or_404(consultation_id)
    if not _can_access_consultation(consult):
        flash('Access denied.', 'danger')
        return redirect(url_for('main.index'))

    messages = consult.messages.all()
    # Mark messages as read
    (ChatMessage.query
     .filter_by(consultation_id=consultation_id, is_read=False)
     .filter(ChatMessage.sender_id != current_user.id)
     .update({'is_read': True}))
    db.session.commit()

    return render_template('chat/room.html',
                           consult=consult, messages=messages)


# ─────────────── Socket.IO events ────────────────────────────────────────────

@socketio.on('join')
def on_join(data):
    room = str(data.get('room'))
    join_room(room)
    emit('status', {'msg': f'{current_user.full_name} joined the room.'}, room=room)


@socketio.on('leave')
def on_leave(data):
    room = str(data.get('room'))
    leave_room(room)
    emit('status', {'msg': f'{current_user.full_name} left the room.'}, room=room)


@socketio.on('send_message')
def handle_message(data):
    room           = str(data.get('room'))
    consultation_id = int(data.get('consultation_id'))
    content        = data.get('message', '').strip()

    if not content:
        return

    msg = ChatMessage(
        consultation_id = consultation_id,
        sender_id       = current_user.id,
        content         = content,
        timestamp       = datetime.utcnow()
    )
    db.session.add(msg)

    # Auto-update consultation status to in_progress
    consult = db.session.get(Consultation, consultation_id)
    if consult and consult.status == 'accepted':
        consult.status = 'in_progress'

    db.session.commit()

    emit('receive_message', msg.to_dict(), room=room)
