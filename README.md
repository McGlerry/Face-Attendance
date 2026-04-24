# Face Recognition Attendance System

A classroom-optimized face recognition attendance system built with Flask, OpenCV, and face-recognition library.

## 🚀 Features

- Real-time face recognition and attendance marking
- Classroom-optimized detection algorithms
- Memory-efficient processing with adaptive frame skipping
- Comprehensive security logging and audit trails
- Web-based dashboard for monitoring and management
- CSV/PDF export capabilities
- Multi-camera support with automatic failover

## 🛡️ Security Features

- Environment-based configuration (no hardcoded secrets)
- HTTPS support for camera access
- Session security with proper cookie settings
- Input validation and sanitization
- Security event logging

## 📋 Prerequisites

- Python 3.8+
- Webcam or IP camera
- Modern web browser (Chrome, Firefox, Edge)

## 🔧 Installation

1. **Clone or download the project**
   ```bash
   cd "c:\dev\gui flask - claude improvement - Final"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your secure secret key
   ```

5. **Run the application**
   ```bash
   # For development
   python main.py

   # For production with HTTPS
   python run_https.py
   ```

## 🔐 Security Configuration

### Environment Variables

Set these environment variables for production:

```bash
FLASK_ENV=production
FLASK_SECRET_KEY=your-very-secure-random-key-here
```

### Secret Key Generation

Generate a secure secret key:

```python
import secrets
print(secrets.token_hex(32))
```

## 📁 Project Structure

```
app/
├── __init__.py          # Flask application factory
├── routes/              # Route blueprints
│   ├── main.py         # Main routes (dashboard, video feed)
│   ├── api.py          # API endpoints
│   ├── students.py     # Student management
│   └── classes.py      # Class management
├── services/           # Business logic
│   ├── database.py     # Database operations
│   └── ...            # Other services
├── utils/              # Utilities
│   ├── config.py       # Configuration management
│   └── memory.py       # Memory optimization
└── templates/          # Jinja2 templates

data/                   # Application data
├── dataset/           # Face images
├── logs/              # Log files
├── exports/           # Exported reports
└── system_config.json # Configuration

static/                # Static assets
├── css/
├── js/
└── uploads/
```

## 🚦 Usage

1. **Start the system**: Access `http://localhost:5000`
2. **Add students and classes** through the web interface
3. **Capture face images** for each student
4. **Train the recognition model**
5. **Start the attendance system**
6. **Monitor live feed** and view attendance records

## 🔍 API Endpoints

### System Control
- `POST /api/start_system` - Start attendance system
- `POST /api/stop_system` - Stop attendance system
- `GET /api/system_status` - Get system status

### Training
- `POST /api/train_model` - Start model training
- `GET /api/training_progress` - Get training progress

### Video Feeds
- `GET /video_feed` - Live video feed
- `GET /api/capture_video_feed` - Face capture feed

## 📊 Configuration

The system uses a JSON configuration file (`data/system_config.json`) with the following sections:

- **Camera Settings**: Resolution, FPS, camera indices
- **Recognition Settings**: Thresholds, detection models, quality validation
- **System Settings**: Auto-restart, logging, security features
- **Classroom Settings**: Capacity, grace periods
- **Display Settings**: Feed quality, UI preferences
- **Performance Settings**: Memory management, processing optimization

## 🔧 Troubleshooting

### Camera Issues
- Check camera permissions in browser
- Ensure HTTPS for camera access (required by browsers)
- Try different camera indices in configuration

### Performance Issues
- Adjust `process_every_nth_frame` in recognition settings
- Enable adaptive processing for memory management
- Check system resources (RAM, CPU)

### Recognition Issues
- Ensure good lighting conditions
- Capture multiple angles of faces during enrollment
- Adjust recognition thresholds in configuration

## 📝 Development

### Code Quality
- Follow PEP 8 style guidelines
- Add type hints for better maintainability
- Write comprehensive error handling

### Testing
- Add unit tests for critical functions
- Test with different camera configurations
- Validate security features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with privacy laws and regulations when deploying in real environments.

## ⚠️ Important Notes

- **Production Deployment**: Always use HTTPS in production
- **Data Privacy**: Handle face images and attendance data responsibly
- **Security**: Regularly update dependencies and monitor for vulnerabilities
- **Backup**: Implement regular backups of the database and configuration

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `data/logs/`
3. Ensure all dependencies are properly installed
4. Verify camera and network permissions