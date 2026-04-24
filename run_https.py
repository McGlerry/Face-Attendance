"""
Run Flask with HTTPS support for camera access over network.
This enables camera access from other devices on your network.
"""
import os
import subprocess
import logging
import socket
from app import create_app

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_cert():
    """Generate a self-signed SSL certificate"""
    cert_file = 'cert.pem'
    key_file = 'key.pem'

    if os.path.exists(cert_file) and os.path.exists(key_file):
        logger.info("SSL certificates already exist")
        return cert_file, key_file

    logger.info("Generating SSL certificates...")
    try:
        result = subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
            '-keyout', key_file, '-out', cert_file,
            '-days', '365', '-nodes',
            '-subj', '/CN=localhost'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("SSL certificates generated successfully")
            return cert_file, key_file
        else:
            logger.error(f"Failed to generate certificates: {result.stderr}")
            return None, None
    except FileNotFoundError:
        logger.error("OpenSSL not found. Please install OpenSSL.")
        logger.info("Manual command: openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'")
        return None, None

def main():
    logger.info("=" * 60)
    logger.info("Face Recognition Attendance System - HTTPS Server")
    logger.info("=" * 60)

    # Generate certificates
    cert_file, key_file = generate_cert()

    if not cert_file:
        logger.error("Please install OpenSSL or generate certificates manually.")
        logger.info("Then run: python run_https.py")
        return

    logger.info("Starting HTTPS server...")
    logger.info("-" * 60)
    logger.info("📍 Access URLs:")
    logger.info("   • On this computer:  https://localhost:5000")
    logger.info("   • On network:        https://<your-ip>:5000")
    logger.info()
    logger.info("⚠️  Note: You'll need to accept the self-signed certificate")
    logger.info("   warning in your browser to access the camera.")
    logger.info("-" * 60)

    # Get local IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        logger.info(f"Your computer's IP address: {local_ip}")
        logger.info(f"   Other devices can access at: https://{local_ip}:5000")
    except Exception:
        pass

    logger.info("Press Ctrl+C to stop the server")

    # Run the app with SSL context
    os.environ['FLASK_ENV'] = 'development'
    app = create_app()
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        ssl_context=(cert_file, key_file),
        threaded=True
    )

if __name__ == '__main__':
    main()

