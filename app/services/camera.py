# app/services/camera.py
import cv2
import time
import threading
import logging

logger = logging.getLogger(__name__)

# =========================
# IMPROVED CAMERA MANAGER (DUAL BUFFERING FOR LOW LAG)
# =========================

class CameraManager:
    """Separated video streaming and face processing with dual buffering"""

    def __init__(self, camera_index=0, config=None, auto_start=False):
        self.camera = None
        self.camera_index = camera_index
        self.config = config
        self.is_running = False

        # Separate frame buffers for dual buffering
        self.display_frame = None  # For video feed (30fps)
        self.processing_frame = None  # For face recognition (5fps)

        self.display_lock = threading.Lock()
        self.processing_lock = threading.Lock()

        # Frame rate control
        self.display_fps = 30  # Smooth video streaming
        self.processing_fps = 5  # Face recognition rate

        self.last_display_time = 0
        self.last_processing_time = 0

        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.restart_attempts = 0
        self.last_restart_time = 0

        # Auto-start camera if enabled
        if auto_start:
            logger.info("Auto-starting camera in monitor mode...")
            self.start_camera()

    def start_camera(self):
        if self.is_running:
            return True

        # Circuit breaker logic - with null safety for config
        if self.config and hasattr(self.config, 'config'):
            system_settings = self.config.config.get('system_settings', {})
            if system_settings.get('auto_restart_camera'):
                max_attempts = system_settings.get('max_restart_attempts', 5)
                restart_delay = system_settings.get('restart_delay_seconds', 10)
                
                if self.restart_attempts >= max_attempts:
                    if (time.time() - self.last_restart_time) < restart_delay:
                        logger.warning(
                            "Camera circuit open. Too many restart attempts.")
                        return False
                    else:
                        logger.info(
                            "Camera circuit closed. Retrying camera start.")
                        self.restart_attempts = 0

        logger.debug(f"Starting live camera on index {self.camera_index}")

        # Try DirectShow backend on Windows for better performance
        self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            # Fallback to default
            self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            # Try backups - with null safety
            backup_indices = []
            if self.config and hasattr(self.config, 'config'):
                backup_indices = self.config.config.get('camera_settings', {}).get('backup_camera_indices', [])
            
            for backup_idx in backup_indices:
                try:
                    self.camera.release()
                except RuntimeError:
                    pass
                self.camera = cv2.VideoCapture(backup_idx, cv2.CAP_DSHOW)
                if self.camera.isOpened():
                    self.camera_index = backup_idx
                    logger.debug(f"Switched to backup camera {backup_idx}")
                    break

            if not self.camera.isOpened():
                logger.exception("Cannot open any webcam for live camera")
                self.restart_attempts += 1
                self.last_restart_time = time.time()
                return False

        # Set properties with better values for streaming
        try:
            if self.config and hasattr(self.config, 'config'):
                cs = self.config.config.get('camera_settings', {})
                if cs:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cs.get('frame_width', 640))
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cs.get('frame_height', 480))
                    self.camera.set(cv2.CAP_PROP_FPS, cs.get('fps', 30))
                    # CRITICAL: Set buffer size to 1 to reduce lag
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Failed to set some camera properties: {e}")

        self.is_running = True
        self.restart_attempts = 0
        self.start_time = time.time()  # Update start_time when camera starts
        threading.Thread(target=self._capture_loop, daemon=True).start()
        logger.info(
            f"Camera successfully started on index {self.camera_index}.")
        return True

    def _capture_loop(self):
        """Optimized capture with dual buffering"""
        consecutive_failures = 0
        max_consecutive_failures = 30

        logger.info("Camera capture loop started (dual buffering)")

        while self.is_running:
            try:
                # Use direct read() instead of grab()/retrieve()
                ret, frame = self.camera.read()

                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures % 10 == 0:  # Log every 10th failure
                        logger.warning(
                            f"Camera read failed ({consecutive_failures} consecutive failures)")

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive read failures")
                        self._handle_camera_failure()
                        break

                    time.sleep(0.1)
                    continue

                # Success! Reset failure counter
                consecutive_failures = 0

                current_time = time.time()

                # Update display frame at 30fps (smooth video)
                if current_time - self.last_display_time >= 1/self.display_fps:
                    with self.display_lock:
                        self.display_frame = frame.copy()
                    self.last_display_time = current_time

                # Update processing frame at 5fps (face recognition - don't block display)
                if current_time - self.last_processing_time >= 1/self.processing_fps:
                    with self.processing_lock:
                        self.processing_frame = frame.copy()
                    self.last_processing_time = current_time

                # Update FPS counter
                self.frame_count += 1
                if current_time - self.last_fps_time >= 1.0:
                    self.fps_counter = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # Minimal sleep to prevent CPU spinning
                time.sleep(0.001)

            except (RuntimeError, ValueError) as e:
                consecutive_failures += 1
                logger.exception(
                    f"Capture loop error (failure #{consecutive_failures}): {e}")

                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive errors in capture loop")
                    self._handle_camera_failure()
                    break

                time.sleep(0.1)

        logger.info("Camera capture loop ended")

    def _handle_camera_failure(self):
        """Handle camera failures"""
        self.is_running = False
        # Safe config access
        auto_restart = False
        restart_delay = 10
        if self.config and hasattr(self.config, 'config'):
            auto_restart = self.config.config.get('system_settings', {}).get('auto_restart_camera', False)
            restart_delay = self.config.config.get('system_settings', {}).get('restart_delay_seconds', 10)
        
        if auto_restart:
            self.restart_attempts += 1
            self.last_restart_time = time.time()
            time.sleep(restart_delay)
            self.start_camera()
        else:
            logger.error("Camera stopped and auto-restart is disabled.")

    def get_display_frame(self):
        """Fast frame retrieval for video streaming (30fps)"""
        with self.display_lock:
            return self.display_frame.copy() if self.display_frame is not None else None

    def get_frame(self):
        """Get current frame for processing"""
        return self.get_display_frame()

    def stop_camera(self):
        if not self.is_running:
            return
        self.is_running = False
        try:
            if self.camera:
                self.camera.release()
        except RuntimeError:
            pass
        self.camera = None
        logger.info("Camera successfully stopped.")
