# app/routes/main.py
from flask import Blueprint, render_template, Response, request, flash, redirect, url_for, send_from_directory, session
import cv2
import numpy as np
import time
import os
from datetime import datetime
from functools import wraps

from .. import config_manager, db_manager, attendance_system

# Constants
EXPORTS_DIR = "data/exports"

main_bp = Blueprint('main', __name__)

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function

# Public route - no login required
@main_bp.route('/')
def index():
    # Check if already logged in
    if 'user_id' in session:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('main.login'))

# Public route - login page
@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    # Check if already logged in
    if 'user_id' in session:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Verify and get user in single call - more efficient
        user = db_manager.verify_user(username, password)
        if user:
            # Set session - use user info returned from verify_user (no redundant query)
            session['user_id'] = user['user_id']
            session['username'] = username
            session['role'] = user.get('role', 'admin')  # Use actual role from database
            session['logged_in'] = True
            
            # Update last login
            db_manager.update_last_login(username)
            
            # Log security event
            db_manager.log_security_event("USER_LOGIN", details=f"User {username} logged in successfully")
            
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

# Logout route
@main_bp.route('/logout')
def logout():
    """Logout and redirect to login page"""
    username = session.get('username', 'Unknown')
    
    # Log security event
    db_manager.log_security_event("USER_LOGOUT", details=f"User {username} logged out")
    
    # Clear session
    session.clear()
    
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.login'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    status = attendance_system.get_system_status()
    students = db_manager.get_all_students()
    classes = db_manager.get_all_classes()
    current_time = datetime.now().strftime("%H:%M:%S")

    # Get count of only the attendance records that were actually marked for today
    attendance_count = db_manager.get_today_attendance_count()

    return render_template('dashboard.html',
                           status=status,
                           students=students,
                           classes=classes,
                           attendance_count=attendance_count,
                           current_time=current_time,
                           body_class='dashboard-bg')

@main_bp.route('/live_monitor')
@login_required
def live_monitor():
    status = attendance_system.get_system_status()
    return render_template('live_monitor.html', status=status, body_class='dashboard-bg')

@main_bp.route('/reports')
@login_required
def reports():
    classes_list = db_manager.get_all_classes()
    return render_template('reports.html', classes=classes_list)

@main_bp.route('/security_logs')
@login_required
def security_logs():
    """View security audit logs"""
    logs = db_manager.get_security_logs(200)  # Get last 200 events
    # Convert Row objects to dictionaries for easier template rendering
    formatted_logs = []
    for log in logs:
        formatted_logs.append(dict(log))
    return render_template('security_logs.html', logs=formatted_logs)

@main_bp.route('/video_feed')
@login_required
def video_feed():
    """Optimized route for live video streaming with proper frame handling"""
    def generate():
        if not attendance_system.camera_manager.is_running:
            # If camera is not active, yield a single black frame to indicate error
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Camera Inactive", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' +
                       str(len(frame_bytes)).encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
            return  # Exit generator early

        last_frame_time = 0
        frame_skip_count = 0

        # Define encode parameters for JPEG compression
        encode_param = [
            cv2.IMWRITE_JPEG_QUALITY,
            config_manager.config['display_settings']['jpeg_quality'],
            cv2.IMWRITE_JPEG_OPTIMIZE, 1  # Enable optimization
        ]

        while True:
            try:
                current_time = time.time()

                # Throttle to ~30fps for web streaming (improved smoothness)
                if current_time - last_frame_time < 0.033:  # ~33ms = 30fps
                    time.sleep(0.01)
                    continue

                frame = attendance_system.get_processed_frame()

                if frame is None:
                    frame_skip_count += 1
                    if frame_skip_count > 30:  # If no frame for 1.5 seconds
                        # Yield a "No signal" frame to keep connection alive and prevent timeouts
                        no_signal_frame = np.zeros(
                            (480, 640, 3), dtype=np.uint8)
                        cv2.putText(no_signal_frame, "No Signal", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        ret, buffer = cv2.imencode(
                            '.jpg', no_signal_frame, encode_param)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            last_frame_time = current_time
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' +
                                   str(len(frame_bytes)).encode() + b'\r\n'
                                   b'\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.05)
                    continue

                frame_skip_count = 0

                ret, buffer = cv2.imencode('.jpg', frame, encode_param)

                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                last_frame_time = current_time

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' +
                       str(len(frame_bytes)).encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')

            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.1)

    response = Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

    # Add headers to prevent caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@main_bp.route('/exports/<filename>')
@login_required
def serve_export(filename):
    """Serve exported files for download"""
    try:
        # Security: Only allow files from EXPORTS_DIR
        safe_path = os.path.join(EXPORTS_DIR, filename)
        if not os.path.exists(safe_path):
            flash('File not found', 'error')
            return redirect(url_for('main.reports'))

        # Determine mimetype
        if filename.endswith('.csv'):
            mimetype = 'text/csv'
        elif filename.endswith('.pdf'):
            mimetype = 'application/pdf'
        elif filename.endswith('.docx'):
            mimetype = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            mimetype = 'application/octet-stream'

        return send_from_directory(
            EXPORTS_DIR,
            filename,
            as_attachment=True,
            mimetype=mimetype,
            download_name=filename
        )
    except Exception:
        flash('Failed to download file', 'error')
        return redirect(url_for('main.reports'))

@main_bp.route('/settings')
@login_required
def settings():
    return render_template('settings.html', config=config_manager.config)

@main_bp.route('/config')
@login_required
def config_page():
    return render_template('config.html', config=config_manager.config)

@main_bp.route('/save_settings', methods=['POST'])
@login_required
def save_settings():
    try:
        # Update camera settings
        config_manager.config['camera_settings']['camera_index'] = int(
            request.form['camera_index'])

        backup_indices_str = request.form.get(
            'backup_camera_indices', '').strip()
        if backup_indices_str:
            try:
                backup_indices = [int(idx.strip()) for idx in backup_indices_str.split(
                    ',') if idx.strip().isdigit()]
                config_manager.config['camera_settings']['backup_camera_indices'] = backup_indices
            except ValueError:
                flash(
                    'Invalid format for backup camera indices. Please use comma-separated numbers.', 'error')
                return redirect(url_for('main.settings'))
        else:
            config_manager.config['camera_settings']['backup_camera_indices'] = [
            ]

        config_manager.config['camera_settings']['frame_width'] = int(
            request.form['frame_width'])
        config_manager.config['camera_settings']['frame_height'] = int(
            request.form['frame_height'])
        config_manager.config['camera_settings']['fps'] = int(
            request.form['fps'])

        # Update recognition settings (enhanced)
        config_manager.config['recognition_settings']['face_recognition_threshold'] = float(
            request.form['face_threshold'])
        config_manager.config['recognition_settings']['confidence_threshold'] = float(
            request.form.get('confidence_threshold', 0.3))
        config_manager.config['recognition_settings']['cooldown_seconds'] = int(
            request.form['cooldown_seconds'])
        config_manager.config['recognition_settings']['process_every_nth_frame'] = int(
            request.form['process_nth_frame'])
        config_manager.config['recognition_settings']['capture_images_per_student'] = int(
            request.form['capture_per_student'])
        config_manager.config['recognition_settings']['require_minimum_detections'] = int(
            request.form.get('min_detections', 5))
        config_manager.config['recognition_settings']['detection_window_seconds'] = int(
            request.form.get('detection_window_seconds', 15))
        config_manager.config['recognition_settings']['min_face_area'] = int(
            request.form.get('min_face_area', 8000))
        config_manager.config['recognition_settings']['max_students_per_frame'] = int(
            request.form.get('max_students_per_frame', 3))

        # Changed to "hog" for faster live feed processing
        # Faster detection for live feed
        config_manager.config['recognition_settings']['detection_model'] = "hog"

        # Update system settings (enhanced)
        config_manager.config['system_settings']['auto_export_enabled'] = 'auto_export' in request.form
        config_manager.config['system_settings']['export_interval_minutes'] = int(
            request.form['export_interval'])
        config_manager.config['system_settings']['enable_quality_validation'] = 'quality_validation' in request.form
        config_manager.config['system_settings']['enable_multiple_confirmations'] = 'multiple_confirmations' in request.form
        config_manager.config['system_settings']['security_logging'] = 'security_logging' in request.form
        config_manager.config['system_settings']['auto_restart_camera'] = 'auto_restart_camera' in request.form
        config_manager.config['system_settings']['max_restart_attempts'] = int(
            request.form.get('max_restart_attempts', 5))
        config_manager.config['system_settings']['restart_delay_seconds'] = int(
            request.form.get('restart_delay_seconds', 10))
        config_manager.config['system_settings']['classroom_mode'] = 'classroom_mode' in request.form

        # Update classroom settings
        config_manager.config['classroom_settings']['attendance_grace_period_minutes'] = int(
            request.form.get('grace_period_minutes', 10))
        config_manager.config['classroom_settings']['auto_mark_present_threshold'] = float(
            request.form.get('auto_mark_threshold', 0.25))
        config_manager.config['classroom_settings']['lighting_validation'] = 'lighting_validation' in request.form

        # Update display settings
        config_manager.config['display_settings']['show_confidence_scores'] = 'show_confidence' in request.form
        config_manager.config['display_settings']['jpeg_quality'] = int(
            request.form.get('jpeg_quality', 80))
        config_manager.config['display_settings']['show_recognition_overlay'] = 'show_overlay' in request.form

        # Save to file
        if config_manager.save_config():
            db_manager.log_security_event(
                "SETTINGS_UPDATED", details="System configuration changed")
            flash('Classroom settings saved successfully. Restart the application for all changes to take effect.', 'success')
        else:
            flash('Failed to save settings.', 'error')

    except Exception as e:
        flash(f'Invalid settings: {str(e)}', 'error')

    return redirect(url_for('main.settings'))