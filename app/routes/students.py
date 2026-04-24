# app/routes/students.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
import os
from functools import wraps

from .. import db_manager, config_manager

student_bp = Blueprint('student', __name__, url_prefix='/students')

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function

@student_bp.route('/')
@login_required
def students():
    students_list = db_manager.get_all_students()
    return render_template('students.html', students=students_list)

@student_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        try:
            student_id = int(request.form['student_id'])
        except Exception:
            flash('Student ID must be an integer.', 'error')
            return redirect(url_for('student.add_student'))

        name = request.form['name'].strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()

        success, message = db_manager.add_student(
            student_id, name, email, phone)
        if success:
            flash('Student added successfully! You can now capture faces for this student from the student list.', 'success')
            return redirect(url_for('student.students'))
        else:
            flash(message, 'error')

    return render_template('add_student.html')

@student_bp.route('/capture_faces/<int:student_id>')
@login_required
def capture_faces(student_id):
    student = db_manager.get_student_by_id(student_id)
    if not student:
        flash('Student not found.', 'error')
        return redirect(url_for('student.students'))

    # Get current capture count for this student - use DATASET_DIR from config
    dataset_dir = config_manager.config['system_settings'].get('dataset_dir', 'data/dataset')
    current_captures = 0
    if os.path.exists(dataset_dir):
        current_captures = len([f for f in os.listdir(dataset_dir)
                               if f.startswith(f"{student_id}_")])

    return render_template('capture_faces.html', student=student, config=config_manager.config, current_captures=current_captures)

@student_bp.route('/update', methods=['POST'])
@login_required
def update_student():
    try:
        student_id = int(request.form['student_id'])
        name = request.form['name'].strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()

        if not name:
            return jsonify({'success': False, 'message': 'Name is required'})

        success, message = db_manager.update_student(
            student_id, name, email, phone)
        if success:
            db_manager.log_security_event(
                "STUDENT_UPDATED", student_id, details=f"Name: {name}")
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@student_bp.route('/delete', methods=['POST'])
@login_required
def delete_student():
    try:
        data = request.get_json()
        student_id = int(data['student_id'])

        success, message = db_manager.delete_student(student_id)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})