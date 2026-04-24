# app/utils/memory.py
import time
import cv2
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

# =========================
# MEMORY OPTIMIZATION SYSTEM
# =========================
class MemoryOptimizedSystem:
    """Monitor and manage system memory"""

    def __init__(self):
        self.memory_threshold = 0.85  # 85% RAM usage
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        self.attendance_system = None  # Reference to attendance system for cleanup

    def check_memory_usage(self):
        """Monitor current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_mb': memory.available / (1024 * 1024),
            'used_mb': memory.used / (1024 * 1024),
            'warning': memory.percent > self.memory_threshold * 100
        }

    def _cleanup_old_buffers(self):
        """Clean up old recognition buffers from attendance system"""
        if self.attendance_system is None:
            logger.debug("Cannot cleanup buffers: attendance system not set")
            return

        try:
            current_time = time.time()
            detection_window = 30  # 30 seconds window

            # Clean recognition buffer - it's a deque of dicts, not a dict
            if hasattr(self.attendance_system, 'recognition_buffer'):
                # Filter out old entries from the deque
                while self.attendance_system.recognition_buffer:
                    oldest_entry = self.attendance_system.recognition_buffer[0]
                    if isinstance(oldest_entry, dict):
                        entry_time = oldest_entry.get('timestamp', 0)
                        if current_time - entry_time > detection_window:
                            # Remove old entry from left side
                            self.attendance_system.recognition_buffer.popleft()
                        else:
                            # Deque is ordered, so if oldest is not expired, none are
                            break
                    else:
                        # Unknown entry type, skip
                        self.attendance_system.recognition_buffer.popleft()

            logger.debug("Recognition buffer cleanup completed")

        except Exception as e:
            logger.exception(f"Error cleaning up buffers: {e}")

    def optimize_image_storage(self, image):
        """
        Compress images before storing:
        - Reduce resolution if too large
        - Apply JPEG compression
        """
        h, w = image.shape[:2]

        # Resize if larger than needed
        max_dimension = 1024
        if w > max_dimension or h > max_dimension:
            scale = max_dimension / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)

        # Encode with compression
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 85,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ]

        _, encoded = cv2.imencode('.jpg', image, encode_params)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)