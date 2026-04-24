# app/routes/classes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
import sqlite3
from functools import wraps

from .. import db_manager

class_bp = Blueprint('class', __name__, url_prefix='/classes')

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function

@class_bp.route('/')
@login_required
def classes():
    classes_list = db_manager.get_all_classes()
    return render_template('classes.html', classes=classes_list)

@class_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add_class():
    if request.method == 'POST':
        class_name = request.form['class_name'].strip()
        class_code = request.form['class_code'].strip()
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        days_of_week = request.form.get('days_of_week', '').strip()

        if not class_name or not class_code:
            flash('Class name and code are required', 'error')
            return render_template('add_class.html')

        success, message = db_manager.add_class(
            class_name, class_code, start_time, end_time, days_of_week)
        if success:
            flash(message, 'success')
            return redirect(url_for('class.classes'))
        else:
            flash(message, 'error')

    return render_template('add_class.html')

@class_bp.route('/update', methods=['POST'])
@login_required
def update_class():
    try:
        class_id = int(request.form['class_id'])
        class_name = request.form['class_name'].strip()
        class_code = request.form['class_code'].strip()
        start_time = request.form.get('start_time', '').strip()
        end_time = request.form.get('end_time', '').strip()
        days_of_week = request.form.get('days_of_week', '').strip()

        if not class_name or not class_code:
            return jsonify({'success': False, 'message': 'Class name and code are required'})

        with db_manager.cursor() as c:
            c.execute("""
                UPDATE classes
                SET class_name = ?, class_code = ?, start_time = ?, end_time = ?, days_of_week = ?
                WHERE class_id = ?
            """, (class_name, class_code, start_time, end_time, days_of_week, class_id))

        return jsonify({'success': True, 'message': 'Class updated successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Class code already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@class_bp.route('/delete', methods=['POST'])
@login_required
def delete_class():
    try:
        data = request.get_json()
        class_id = int(data['class_id'])

        with db_manager.cursor() as c:
            # ON DELETE CASCADE should handle attendance and enrollment, but explicit is safer
            c.execute("DELETE FROM attendance WHERE class_id = ?", (class_id,))
            c.execute(
                "DELETE FROM class_enrollment WHERE class_id = ?", (class_id,))
            c.execute("DELETE FROM classes WHERE class_id = ?", (class_id,))

        return jsonify({'success': True, 'message': 'Class deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@class_bp.route('/manage_enrollment/<int:class_id>')
@login_required
def manage_enrollment(class_id):
    selected_class = db_manager.get_class_by_id(class_id)
    if not selected_class:
        flash('Class not found.', 'error')
        return redirect(url_for('class.classes'))

    enrolled_students = db_manager.get_enrolled_students(class_id)
    unenrolled_students = db_manager.get_unenrolled_students(class_id)

    return render_template('manage_enrollment.html',
                           selected_class=selected_class,
                           enrolled_students=enrolled_students,
                           unenrolled_students=unenrolled_students)

@class_bp.route('/update_enrollment', methods=['POST'])
@login_required
def update_enrollment():
    class_id = request.form.get('class_id', type=int)
    student_id = request.form.get('student_id', type=int)
    action = request.form.get('action')  # 'enroll' or 'unenroll'

    if not class_id or not student_id or not action:
        return jsonify({'success': False, 'message': 'Missing data.'})

    if action == 'enroll':
        success, message = db_manager.enroll_student(student_id, class_id)
    elif action == 'unenroll':
        success, message = db_manager.unenroll_student(student_id, class_id)
    else:
        return jsonify({'success': False, 'message': 'Invalid action.'})

    return jsonify({'success': success, 'message': message})