from flask import Blueprint, render_template, redirect, url_for, jsonify
from flask_login import current_user
from sqlalchemy import text
from app import db
from app.models.user import DoctorProfile

main_bp = Blueprint('main', __name__)


@main_bp.route('/health')
def health():
    """Report ready only when the app can reach its database."""
    try:
        db.session.execute(text('SELECT 1'))
        return jsonify(status='ok'), 200
    except Exception:
        db.session.rollback()
        return jsonify(status='unavailable'), 503


@main_bp.route('/')
def index():
    doctors = DoctorProfile.query.filter_by(is_available=True).limit(6).all()
    return render_template('index.html', doctors=doctors)


@main_bp.route('/about')
def about():
    return render_template('about.html')


@main_bp.route('/awareness')
def awareness():
    return render_template('awareness.html')


@main_bp.route('/career')
def career():
    return render_template('career.html')


@main_bp.route('/what-is-cancer')
def what_is_cancer():
    return render_template('what_is_cancer.html')


@main_bp.route('/what-is-breast-cancer')
def what_is_breast_cancer():
    return render_template('what_is_breast_cancer.html')


@main_bp.route('/image-database')
def image_database():
    return render_template('image_database.html')


@main_bp.route('/blog')
def blog():
    return render_template('blog.html')


@main_bp.route('/contact')
def contact():
    return render_template('contact.html')


@main_bp.route('/project')
def project():
    return render_template('project.html')
