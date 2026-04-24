"""
Attendance system service for managing face recognition and attendance tracking.
Handles system lifecycle, frame processing, and attendance marking.
Enhanced with liveness detection and face alignment support.
"""

import time
import threading
import logging
import sqlite3
import cv2
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class ClassroomAttendanceSystem:
    """
    Enhanced classroom attendance system with optimized face recognition,
    quality validation, tiered confidence matching, liveness detection,
    and face alignment for improved security and accuracy.
    """

    def __init__(self, config_manager, db_manager, face_model, camera_manager, memory_optimizer):
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.face_model = face_model
        self.camera_manager = camera_manager
        self.memory_optimizer = memory_optimizer

        # System state
        self.running = False
        self.processing_thread = None
        self.last_frame_time = 0
        self.frame_count = 0

        # Recognition state
        self.recognition_buffer = deque(maxlen=50)  # Store recent recognitions
        self.student_cooldowns = {}  # Track cooldown periods
        self.active_classes = []  # Currently active classes
        
        # Liveness tracking per student
        self.liveness_tracking = {}  # Track liveness detection per student

        # Performance tracking
        self.stats = {
            'faces_detected': 0,
            'recognitions': 0,
            'attendances_marked': 0,
            'camera_fps': 0.0,
            'faces_rejected_quality': 0,
            'faces_rejected_liveness': 0,  # NEW: Track liveness failures
            'auto_mark_matches': 0,
            'high_confidence_matches': 0,
            'standard_matches': 0,
            'suspicious_matches': 0,
            'processing_fps': 0.0,
            'average_processing_time_ms': 0.0
        }

        # Processing timing
        self.processing_times = deque(maxlen=100)

        logger.info("ClassroomAttendanceSystem initialized with liveness detection")

    def start_system(self):
        """Start the attendance system"""
        if self.running:
            return True, "System already running"

        try:
            logger.info("Starting Classroom Attendance System...")

            # Start camera if not already running
            if not self.camera_manager.is_running:
                camera_success = self.camera_manager.start_camera()
                if not camera_success:
                    return False, "Failed to start camera"

            # Reset stats
            self._reset_stats()

            # Start processing thread
            self.running = True
            self.processing_thread = threading.Thread(
                target=self.processing_loop, daemon=True)
            self.processing_thread.start()

            self.db_manager.log_security_event(
                "SYSTEM_STARTED", details="Attendance system started successfully")

            logger.info("Classroom Attendance System started successfully")
            return True, "System started successfully"

        except (threading.ThreadError, RuntimeError) as e:
            logger.exception("Error starting attendance system")
            self.running = False
            return False, f"Failed to start system: {str(e)}"

    def stop_system(self):
        """Stop the attendance system"""
        if not self.running:
            return

        logger.info("Stopping Classroom Attendance System...")
        self.running = False

        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

        # Stop camera
        self.camera_manager.stop_camera()

        self.db_manager.log_security_event(
            "SYSTEM_STOPPED", details="Attendance system stopped")

        logger.info("Classroom Attendance System stopped")

    def processing_loop(self):
        """Main processing loop for face recognition and attendance marking"""
        logger.info("Processing loop started")

        frame_skip_counter = 0
        last_stats_time = time.time()

        while self.running:
            try:
                start_time = time.time()

                # Throttle processing to configured frame rate
                process_every_nth = self.config_manager.config['recognition_settings']['process_every_nth_frame']
                frame_skip_counter += 1

                if frame_skip_counter < process_every_nth:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    continue

                frame_skip_counter = 0

                # Get frame from camera
                frame = self.camera_manager.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Process frame for faces
                self._process_frame(frame)

                # Update performance stats periodically
                current_time = time.time()
                if current_time - last_stats_time >= 5.0:  # Every 5 seconds
                    self._update_performance_stats()
                    last_stats_time = current_time

                # Track processing time
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)

            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                time.sleep(0.1)

        logger.info("Processing loop ended")

    def _process_frame(self, frame):
        """Process a single frame for face recognition"""
        try:
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            detection_model = self.config_manager.config['recognition_settings']['detection_model']
            face_locations = self.face_model.detect_faces(rgb_frame, model=detection_model)

            if not face_locations:
                return

            self.stats['faces_detected'] += len(face_locations)

            # Process each detected face
            for face_location in face_locations:
                self._process_single_face(rgb_frame, face_location, frame)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(f"Error processing frame: {e}")

    def _process_single_face(self, rgb_frame, face_location, original_frame):
        """Process a single detected face with liveness and alignment"""
        try:
            top, right, bottom, left = face_location

            # Quality validation if enabled
            if self.config_manager.config['system_settings']['enable_quality_validation']:
                quality_ok, quality_msg = self.face_model.validate_classroom_face_quality(
                    face_location, original_frame)
                if not quality_ok:
                    self.stats['faces_rejected_quality'] += 1
                    return

            # LIVENESS DETECTION: Check if face is live (not a photo)
            liveness_enabled = self.config_manager.config['recognition_settings'].get('liveness_detection', True)
            if liveness_enabled:
                is_live, liveness_msg = self.face_model.check_liveness(rgb_frame, face_location)
                
                # If liveness check definitively failed, reject this face
                if is_live is False:
                    self.stats['faces_rejected_liveness'] += 1
                    logger.warning(f"Liveness check failed: {liveness_msg}")
                    return
                
                # If still collecting data, continue but log
                if is_live is None:
                    logger.debug(f"Liveness collecting: {liveness_msg}")

            # Get face encoding with alignment (new method returns tuple)
            encoding_result = self.face_model.get_face_encoding(rgb_frame, face_location, use_alignment=True)
            
            # Handle both old and new return formats
            if isinstance(encoding_result, tuple):
                face_encoding, landmarks = encoding_result
            else:
                face_encoding = encoding_result
                landmarks = None
                
            if face_encoding is None:
                return

            # Find matches
            frame_quality = self.face_model.enhanced_system.calculate_frame_quality(face_location, original_frame)
            matches = self.face_model.find_matches(face_encoding, frame_quality)
            if not matches:
                return

            # Process best match
            best_match = matches[0]
            student_id = best_match['student_id']
            confidence = best_match['confidence']

            # Check cooldown
            if self._is_on_cooldown(student_id):
                return

            # Mark attendance based on confidence and settings
            self._mark_attendance_if_qualified(student_id, confidence, best_match['match_type'])

            # Update recognition buffer
            self.recognition_buffer.append({
                'student_id': student_id,
                'confidence': confidence,
                'timestamp': time.time(),
                'match_type': best_match['match_type'],
                'liveness_confirmed': liveness_enabled  # Track if liveness was checked
            })

            self.stats['recognitions'] += 1

        except (RuntimeError, ValueError, TypeError, IndexError) as e:
            logger.exception(f"Error processing single face: {e}")

    def _mark_attendance_if_qualified(self, student_id, confidence, match_type):
        """Mark attendance if confidence meets thresholds"""
        try:
            settings = self.config_manager.config['recognition_settings']

            # Determine if we should mark attendance
            should_mark = False

            if confidence >= settings['face_recognition_threshold']:
                if match_type == 'auto_mark':
                    should_mark = True
                    self.stats['auto_mark_matches'] += 1
                elif confidence >= 0.8:  # High confidence
                    should_mark = True
                    self.stats['high_confidence_matches'] += 1
                elif confidence >= settings['face_recognition_threshold']:
                    should_mark = True
                    self.stats['standard_matches'] += 1
            else:
                self.stats['suspicious_matches'] += 1

            if should_mark:
                # Only mark attendance for currently scheduled classes (active classes)
                active_class_ids = [c['id'] for c in self.active_classes]
                
                if not active_class_ids:
                    logger.debug(f"No active classes - attendance not marked for student {student_id}")
                    return

                # Mark attendance for active classes only
                for class_id in active_class_ids:
                    if self.db_manager.is_student_enrolled(student_id, class_id):
                        success, msg = self.db_manager.mark_attendance(
                            student_id, class_id, confidence, match_type)
                        if success:
                            self.stats['attendances_marked'] += 1
                            self._set_cooldown(student_id)
                            logger.info(f"Attendance marked for student {student_id} in class {class_id} "
                                      f"(confidence: {confidence:.3f}, type: {match_type})")

        except (sqlite3.Error, sqlite3.IntegrityError, ValueError) as e:
            logger.exception(f"Error marking attendance: {e}")

    def _is_on_cooldown(self, student_id):
        """Check if student is on cooldown"""
        if student_id not in self.student_cooldowns:
            return False

        cooldown_end = self.student_cooldowns[student_id]
        return time.time() < cooldown_end

    def _set_cooldown(self, student_id):
        """Set cooldown for student"""
        cooldown_seconds = self.config_manager.config['recognition_settings']['cooldown_seconds']
        self.student_cooldowns[student_id] = time.time() + cooldown_seconds

    def _reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'faces_detected': 0,
            'recognitions': 0,
            'attendances_marked': 0,
            'camera_fps': 0.0,
            'faces_rejected_quality': 0,
            'faces_rejected_liveness': 0,
            'auto_mark_matches': 0,
            'high_confidence_matches': 0,
            'standard_matches': 0,
            'suspicious_matches': 0,
            'processing_fps': 0.0,
            'average_processing_time_ms': 0.0
        }
        self.processing_times.clear()

    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            # Camera FPS
            if self.camera_manager.fps_counter > 0:
                self.stats['camera_fps'] = self.camera_manager.fps_counter

            # Processing FPS
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                self.stats['average_processing_time_ms'] = avg_processing_time
                if avg_processing_time > 0:
                    self.stats['processing_fps'] = 1000 / avg_processing_time

        except (ZeroDivisionError, AttributeError) as e:
            logger.exception(f"Error updating performance stats: {e}")

    def get_processed_frame(self):
        """Get a processed frame with overlays for display"""
        try:
            frame = self.camera_manager.get_frame()
            if frame is None:
                return None

            # Add system status overlay
            self._add_frame_overlay(frame)

            return frame

        except Exception as e:
            logger.exception(f"Error getting processed frame: {e}")
            return None

    def _add_frame_overlay(self, frame):
        """Add status overlay to frame"""
        try:
            height, width = frame.shape[:2]

            # System status
            status_text = f"System: {'Running' if self.running else 'Stopped'}"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Camera status
            camera_text = f"Camera: {'Active' if self.camera_manager.is_running else 'Inactive'}"
            cv2.putText(frame, camera_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Performance stats
            fps_text = f"FPS: {self.stats['processing_fps']:.1f}"
            cv2.putText(frame, fps_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Liveness detection status
            liveness_enabled = self.config_manager.config['recognition_settings'].get('liveness_detection', True)
            liveness_text = f"Liveness: {'ON' if liveness_enabled else 'OFF'}"
            cv2.putText(frame, liveness_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Active classes
            if self.active_classes:
                classes_text = f"Active Classes: {len(self.active_classes)}"
                cv2.putText(frame, classes_text, (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        except Exception as e:
            logger.exception(f"Error adding frame overlay: {e}")

    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            # Update active classes based on current time
            self._update_active_classes()

            # Get recent attendance events
            recent_events = self._get_recent_attendance_events()

            status = {
                'running': self.running,
                'camera_active': self.camera_manager.is_running if self.camera_manager else False,
                'model_loaded': len(self.face_model.known_face_encodings) > 0 if self.face_model else False,
                'recognition_buffer_size': len(self.recognition_buffer),
                'stats': dict(self.stats.copy(), recent_attendance_events=recent_events),
                'active_classes': self.active_classes.copy(),
                'security_features': {
                    'quality_validation': self.config_manager.config['system_settings']['enable_quality_validation'],
                    'multiple_confirmations': self.config_manager.config['system_settings']['enable_multiple_confirmations'],
                    'security_logging': self.config_manager.config['system_settings']['security_logging'],
                    'liveness_detection': self.config_manager.config['recognition_settings'].get('liveness_detection', True),
                    'face_alignment': self.config_manager.config['recognition_settings'].get('face_alignment', True)
                }
            }

            # Add error info if system not running properly
            if not self.running:
                status['error'] = 'System is not running'
            elif not self.camera_manager.is_running:
                status['error'] = 'Camera is not active'
            elif not status['model_loaded']:
                status['error'] = 'Face recognition model not loaded'

            return status

        except Exception as e:
            logger.exception("Error getting system status")
            return {
                'running': False,
                'camera_active': False,
                'error': str(e),
                'stats': self.stats.copy(),
                'active_classes': [],
                'model_loaded': False,
                'recognition_buffer_size': 0,
                'security_features': {
                    'quality_validation': False,
                    'multiple_confirmations': False,
                    'security_logging': False,
                    'liveness_detection': False,
                    'face_alignment': False
                }
            }

    def _update_active_classes(self):
        """Update the list of currently active classes"""
        try:
            current_time = datetime.now()
            current_day = current_time.strftime('%A').lower()
            active_classes = []
            all_classes = self.db_manager.get_all_classes()

            for class_info in all_classes:
                # Check if class has valid time values
                if not class_info.get('start_time') or not class_info.get('end_time'):
                    continue

                # Check if class is scheduled for today
                if class_info['days_of_week'] and current_day not in class_info['days_of_week'].lower():
                    continue

                # Parse time strings with error handling
                try:
                    start_time = datetime.strptime(class_info['start_time'], '%H:%M').time()
                    end_time = datetime.strptime(class_info['end_time'], '%H:%M').time()
                except (ValueError, TypeError):
                    logger.warning(f"Invalid time format for class {class_info.get('class_name', 'Unknown')}")
                    continue

                current_time_obj = current_time.time()

                # Add grace period
                grace_minutes = self.config_manager.config['classroom_settings']['attendance_grace_period_minutes']
                start_with_grace = (datetime.combine(current_time.date(), start_time) -
                                  timedelta(minutes=grace_minutes)).time()

                if start_with_grace <= current_time_obj <= end_time:
                    active_classes.append({
                        'id': class_info['class_id'],
                        'name': class_info['class_name'],
                        'code': class_info['class_code'],
                        'start_time': class_info['start_time'],
                        'end_time': class_info['end_time']
                    })

            self.active_classes = active_classes

        except Exception as e:
            logger.exception("Error updating active classes")
            self.active_classes = []

    def _get_recent_attendance_events(self, limit=10):
        """Get recent attendance events for display on live monitor"""
        try:
            # Use today's date to show only current day attendance
            today_date = datetime.now().strftime('%Y-%m-%d')

            with self.db_manager.cursor() as c:
                # Get recent attendance records with student and class info for today only
                c.execute("""
                    SELECT a.timestamp, s.name, c.class_name, a.confidence_score, a.match_type
                    FROM attendance a
                    JOIN students s ON a.student_id = s.student_id
                    JOIN classes c ON a.class_id = c.class_id
                    WHERE a.date = ?
                    ORDER BY a.timestamp DESC
                    LIMIT ?
                """, (today_date, limit))

                events = []
                for row in c.fetchall():
                    # Format timestamp for display
                    timestamp_obj = datetime.fromisoformat(row['timestamp'])
                    formatted_timestamp = timestamp_obj.strftime('%H:%M:%S')

                    events.append({
                        'name': row['name'],
                        'timestamp': formatted_timestamp,
                        'class_name': row['class_name'],
                        'confidence': row['confidence_score'] or 0.0,
                        'match_type': row['match_type'] or 'standard'
                    })

                return events

        except Exception as e:
            logger.exception(f"Error getting recent attendance events: {e}")
            return []

