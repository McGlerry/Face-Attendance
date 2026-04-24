# app/__init__.py
import os
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Global instances (consider using Flask application context in production)
config_manager = None
db_manager = None
attendance_system = None



def create_app():
    """Application factory pattern for better testability and configuration"""
    global config_manager, db_manager, attendance_system

    # Get the directory containing this file (app/) and go up one level for templates
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)

    # SECURITY: Use environment variable for secret key
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # SECURITY: Configure session settings - use secure cookies only in production (HTTPS)
    is_production = os.environ.get('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_SECURE'] = is_production  # Only use secure cookies in production
    app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookies
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

    app.config['UPLOAD_FOLDER'] = 'static/uploads'

    # Initialize rate limiter
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )

    # Initialize core services

    from .utils.config import SystemConfig
    from .utils.memory import MemoryOptimizedSystem
    from .services.database import OptimizedDBManager
    from .services.face_recognition import EnhancedRecognitionSystem, ClassroomOptimizedFaceModel
    from .services.camera import CameraManager
    from .services.attendance_system import ClassroomAttendanceSystem

    config_manager = SystemConfig()
    memory_optimizer = MemoryOptimizedSystem()

    db_manager = OptimizedDBManager(config_manager)

    # Initialize face recognition system
    face_recognition_system = EnhancedRecognitionSystem(config_manager)
    face_model = ClassroomOptimizedFaceModel(config_manager)
    face_model.enhanced_system = face_recognition_system  # Connect them

    # Initialize camera manager
    camera_manager = CameraManager(
        camera_index=config_manager.config['camera_settings']['camera_index'],
        config=config_manager
    )

    # Initialize attendance system
    attendance_system = ClassroomAttendanceSystem(
        config_manager, db_manager, face_model, camera_manager, memory_optimizer)

    # Connect memory optimizer to attendance system for buffer cleanup
    memory_optimizer.attendance_system = attendance_system

    # Initialize training progress
    from .services.training import reset_training_progress
    reset_training_progress()

    # Create default admin user if not exists
    db_manager.create_default_admin()

    # Register blueprints
    from .routes.main import main_bp
    from .routes.api import api_bp
    from .routes.students import student_bp
    from .routes.classes import class_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(class_bp)

    # Store limiter in app extensions for access in routes
    app.extensions = {'limiter': limiter}

    return app
