# main.py - Entry point for the Flask application
import os
import webbrowser
from threading import Timer
import logging

from app import create_app

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    app = create_app()

    # Register error handlers
    @app.errorhandler(400)
    def bad_request_error(error):
        logger.error(f"Bad Request (400): {error}")
        from flask import render_template
        return render_template('400.html'), 400

    @app.errorhandler(403)
    def forbidden_error(error):
        logger.error(f"Forbidden (403): {error}")
        from flask import render_template
        return render_template('403.html'), 403

    @app.errorhandler(404)
    def not_found_error(error):
        logger.error(f"Not Found (404): {error}")
        from flask import render_template
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.exception(f"Internal Server Error (500): {error}")
        from flask import render_template
        return render_template('500.html'), 500

    def open_browser():
        webbrowser.open_new('http://localhost:5000')

    # Create templates directory placeholders (templates must exist separately)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    logger.info(
        "Starting Classroom-Optimized Flask Face Recognition Attendance System...")
    logger.info(
        "Features: Classroom-specific quality validation, Tiered recognition, Performance optimization, Security logging")
    logger.info("Navigate to http://localhost:5000 to access the application")

    # SECURITY: Only enable debug mode in development
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    if debug_mode:
        logger.info("Running in DEBUG mode (development only)")

    # Open browser after a short delay (only on first run, not on reloader)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        Timer(1, open_browser).start()

    app.run(debug=debug_mode, host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    main()