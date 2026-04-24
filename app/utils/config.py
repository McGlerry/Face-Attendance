# app/utils/config.py
import os
import json
import logging

logger = logging.getLogger(__name__)

# =========================
# PATHS & CONSTANTS
# =========================

# Use absolute paths based on this file's location
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
DB_FILE = os.path.join(BASE_DIR, "attendance_system.db")
LABELS_FILE = os.path.join(BASE_DIR, "labels.pickle")
CONFIG_FILE = os.path.join(BASE_DIR, "system_config.json")

# Create all required directories at startup
DIRECTORIES = [BASE_DIR, DATASET_DIR, LOGS_DIR, EXPORTS_DIR]
for d in DIRECTORIES:
    os.makedirs(d, exist_ok=True)

# Also ensure DB_FILE's directory exists
db_dir = os.path.dirname(DB_FILE)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

# =========================
# CLASSROOM-OPTIMIZED CONFIGURATION
# =========================


class SystemConfig:
    def __init__(self):
        self.default_config = {
            # ======================
            # 📹 CAMERA SETTINGS
            # ======================
            "camera_settings": {
                "camera_index": 0,
                "backup_camera_indices": [1, 2, 3],
                "frame_width": 640,       # Increased for better video quality
                "frame_height": 480,      # Increased for better video quality
                "fps": 30,                # Increased to 30fps for smoother feed
                "buffer_size": 1
            },

            # ======================
            # 🧠 RECOGNITION SETTINGS
            # ======================
            "recognition_settings": {
                "face_recognition_threshold": 0.4,
                "confidence_threshold": 0.3,
                "cooldown_seconds": 30,
                "process_every_nth_frame": 2,          # ↓ Process every 2 frames instead of 5
                "require_minimum_detections": 5,
                "detection_window_seconds": 15,
                "capture_images_per_student": 10,
                # ✅ Changed from "cnn" → "hog" (faster detection)
                "detection_model": "hog",
                "capture_detection_model": "hog",      # ✅ Use HOG for faster face capture
                "training_detection_model": "cnn",    # Use CNN for training (more accurate)
                "min_face_area": 8000,
                "edge_buffer_pixels": 30,
                "detection_window_frames": 5,
                "max_recognition_distance": 0.55,
                "suspicious_match_threshold": 0.40,
                "max_students_per_frame": 3,
                "num_jitters": 1,                    # Number of jitters for face encoding
                # NEW: Liveness detection settings
                "liveness_detection": True,          # Enable eye blink detection
                "liveness_blink_threshold": 0.2,     # Eye aspect ratio threshold
                "liveness_min_blinks": 1,            # Minimum blinks to confirm liveness
                "liveness_max_frames": 30,          # Frames to collect for liveness check
                # NEW: Face alignment settings
                "face_alignment": True,              # Enable face alignment for better accuracy
                "alignment_desired_width": 256,    # Target face width after alignment
            },

            # ======================
            # ⚙️ SYSTEM SETTINGS
            # ======================
            "system_settings": {
                "dataset_dir": os.path.join(BASE_DIR, "dataset"),
                "auto_restart_camera": True,
                "max_restart_attempts": 5,
                "restart_delay_seconds": 10,
                "health_check_interval": 300,
                "cleanup_logs_days": 30,
                "frame_timeout_seconds": 3,
                "auto_export_enabled": True,
                "export_interval_minutes": 30,
                "enable_quality_validation": True,
                "enable_multiple_confirmations": True,
                "security_logging": True,
                "classroom_mode": True
            },

            # ======================
            # 🏫 CLASSROOM SETTINGS
            # ======================
            "classroom_settings": {
                "max_capacity": 50,
                "attendance_grace_period_minutes": 10,
                "auto_mark_present_threshold": 0.25,
                "lighting_validation": True
            },

            # ======================
            # 📺 DISPLAY SETTINGS
            # ======================
            "display_settings": {
                "show_live_feed": True,
                "feed_width": 640,
                "feed_height": 480,
                "capture_feed_width": 480,
                "capture_feed_height": 360,
                "jpeg_quality": 85,       # Optimized for performance
                "show_confidence_scores": True
            },

            # ======================
            # ⚡ PERFORMANCE SETTINGS
            # ======================
            "performance_settings": {
                "max_concurrent_processing": 2,
                "frame_skip_on_high_load": True,
                "adaptive_processing": True,
                "memory_cleanup_interval": 300
            }
        }
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)
                logger.debug("Loaded system_config.json")
                return config
            except (json.JSONDecodeError, OSError) as e:
                logger.exception(f"Error loading config: {e}")

        self.save_config(self.default_config)
        return self.default_config.copy()

    def save_config(self, config=None):
        if config is None:
            config = self.config
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            logger.debug("Saved system_config.json")
            return True
        except OSError as e:
            logger.exception(f"Error saving config: {e}")
            return False

    def _deep_update(self, base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
