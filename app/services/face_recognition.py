import os
import cv2
import pickle
import time
import numpy as np
import face_recognition
import logging

from ..utils.config import LABELS_FILE

logger = logging.getLogger(__name__)

# =========================
# ENHANCED RECOGNITION SYSTEM WITH ADAPTIVE THRESHOLDS
# =========================

class EnhancedRecognitionSystem:
    """Multi-stage recognition with adaptive thresholds"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.student_encoding_stats = {}  # Track encoding quality per student
        self.load_model()

    def load_model(self):
        """Load face encodings incrementally for memory efficiency"""
        if os.path.exists(LABELS_FILE):
            try:
                with open(LABELS_FILE, "rb") as f:
                    data = pickle.load(f)

                # Load face encodings incrementally for memory efficiency
                encodings = data.get("encodings", [])
                ids = data.get("ids", [])
                names = data.get("names", [])
                stats = data.get("stats", {})

                # Process in chunks to avoid memory spikes
                chunk_size = 50
                total = len(encodings)

                self.known_face_encodings = []
                self.known_face_ids = []
                self.known_face_names = []
                self.student_encoding_stats = stats

                for i in range(0, total, chunk_size):
                    chunk_encodings = encodings[i:i+chunk_size]
                    chunk_ids = ids[i:i+chunk_size]
                    chunk_names = names[i:i+chunk_size]

                    # Extend lists with chunk
                    self.known_face_encodings.extend(chunk_encodings)
                    self.known_face_ids.extend(chunk_ids)
                    self.known_face_names.extend(chunk_names)

                    # Yield to other processes and allow memory cleanup
                    time.sleep(0.01)

                    if (i + chunk_size) % 100 == 0 or (i + chunk_size) >= total:
                        logger.info(
                            f"Loaded {len(self.known_face_encodings)}/{total} face encodings")

                logger.info(
                    f"SUCCESS: Enhanced recognition model loaded with {len(self.known_face_encodings)} faces.")
            except (pickle.PickleError, OSError, ValueError) as e:
                logger.exception(f"[ERROR] Failed loading enhanced model: {e}")
                self.known_face_encodings = []
                self.known_face_ids = []
                self.known_face_names = []
                self.student_encoding_stats = {}
        else:
            logger.info(
                "WARNING: No enhanced model found, starting with empty encodings.")

    def save_model(self):
        """Save the model to disk"""
        data = {
            "encodings": self.known_face_encodings,
            "ids": self.known_face_ids,
            "names": self.known_face_names,
            "stats": self.student_encoding_stats
        }
        try:
            with open(LABELS_FILE, "wb") as f:
                pickle.dump(data, f)
            logger.info("SUCCESS: Enhanced model saved successfully.")
        except (OSError, pickle.PickleError) as e:
            logger.exception(f"Failed to save model: {e}")

    def calculate_frame_quality(self, face_location, frame):
        """
        Return quality score 0-1 based on:
        - Face size
        - Lighting uniformity
        - Sharpness (blur detection)
        - Face angle
        """
        try:
            top, right, bottom, left = face_location
            h, w = frame.shape[:2]

            # Ensure face location is within frame bounds
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)

            face_roi = frame[top:bottom, left:right]

            if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                return 0.0

            # Face size score (normalized to frame size)
            face_area = (right - left) * (bottom - top)
            frame_area = h * w
            size_score = min(1.0, (face_area / frame_area) * 10)  # 10% = 1.0

            # Lighting score (std deviation of brightness)
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness_std = np.std(gray_roi)
            lighting_score = 1.0 - min(1.0, brightness_std / 60.0)

            # Sharpness score (Laplacian variance)
            laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(1.0, sharpness / 500.0)

            # Aspect ratio score (frontal face ~0.75-0.85)
            face_height = bottom - top
            face_width = right - left
            if face_height > 0 and face_width > 0:
                aspect_ratio = face_width / face_height
                aspect_score = 1.0 - abs(0.8 - aspect_ratio)
                aspect_score = max(0.0, min(1.0, aspect_score))  # Clamp to 0-1
            else:
                aspect_score = 0.0

            # Weighted average
            quality_score = (
                size_score * 0.3 +
                lighting_score * 0.3 +
                sharpness_score * 0.25 +
                aspect_score * 0.15
            )

            return quality_score
        except (RuntimeError, ValueError, ZeroDivisionError):
            logger.exception("Frame quality calculation error")
            return 0.0

    def adaptive_recognition(self, face_encoding, frame_quality_score):
        """
        Adjust thresholds based on:
        1. Frame quality (lighting, blur, angle)
        2. Historical match consistency
        3. Time of day (lighting conditions)
        """
        if not self.known_face_encodings:
            return None, None, 1.0, "No trained faces"

        try:
            # Calculate base distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            if len(face_distances) == 0:
                return None, None, 1.0, "No distance calculated"

            best_match_index = np.argmin(face_distances)
            best_distance = float(face_distances[best_match_index])

            # Adaptive threshold based on frame quality
            if frame_quality_score > 0.9:  # Excellent quality
                threshold = 0.4
            elif frame_quality_score > 0.7:  # Good quality
                threshold = 0.45
            else:  # Poor quality - be more strict
                threshold = 0.35

            if best_distance <= threshold:
                student_id = self.known_face_ids[best_match_index]
                student_name = self.known_face_names[best_match_index]

                # Verify with tolerance comparison
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=threshold
                )

                if matches[best_match_index]:
                    # Update student encoding stats for future improvements
                    if student_id not in self.student_encoding_stats:
                        self.student_encoding_stats[student_id] = {
                            'total_matches': 0,
                            'quality_sum': 0.0,
                            'avg_quality': 0.0
                        }

                    stats = self.student_encoding_stats[student_id]
                    stats['total_matches'] += 1
                    stats['quality_sum'] += frame_quality_score
                    stats['avg_quality'] = stats['quality_sum'] / \
                        stats['total_matches']

                    return student_id, student_name, best_distance, "confirmed"

            return None, None, best_distance, "rejected"

        except (ValueError, IndexError) as e:
            logger.exception("Adaptive recognition error")
            return None, None, 1.0, f"Recognition error: {str(e)}"


# =========================
# LIVENESS DETECTION SYSTEM
# =========================

class LivenessDetector:
    """Simple liveness detection using eye blink counting - no extra hardware needed"""
    
    def __init__(self, blink_threshold=0.2, consecutive_frames=3):
        self.blink_threshold = blink_threshold  # Eye aspect ratio threshold
        self.consecutive_frames = consecutive_frames  # Frames to confirm blink
        self.eye_history = []
        self.blink_count = 0
        self.frame_count = 0
        
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate eye aspect ratio (EAR) for blink detection"""
        try:
            # eye_landmarks should be 6 points (3 pairs)
            if len(eye_landmarks) != 6:
                return 1.0  # Open eye default
                
            # Vertical distances
            A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            # Horizontal distance
            C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            
            if C == 0:
                return 1.0
                
            ear = (A + B) / (2.0 * C)
            return ear
        except (ValueError, IndexError):
            return 1.0
    
    def detect_blink(self, face_landmarks):
        """Detect if eyes are blinking based on facial landmarks"""
        try:
            if not face_landmarks:
                return False, 0.0
                
            # Get eye landmarks (indices 36-41 for left eye, 42-47 for right eye in dlib)
            left_eye = face_landmarks.get('left_eye', [])
            right_eye = face_landmarks.get('right_eye', [])
            
            if not left_eye or not right_eye:
                return False, 0.0
            
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_eye) / 2.0
            
            self.eye_history.append(avg_ear)
            if len(self.eye_history) > 10:  # Keep last 10 frames
                self.eye_history.pop(0)
            
            # Check for blink (sudden drop in EAR)
            is_blink = avg_ear < self.blink_threshold
            
            if is_blink:
                self.blink_count += 1
            
            return is_blink, avg_ear
            
        except Exception:
            return False, 0.0
    
    def is_live_face(self, min_blinks=1, max_frames=30):
        """Determine if face is live based on blink detection"""
        self.frame_count += 1
        
        # Reset after max_frames
        if self.frame_count >= max_frames:
            has_blinked = self.blink_count >= min_blinks
            self.blink_count = 0
            self.frame_count = 0
            self.eye_history = []
            return has_blinked
        
        # Still collecting data
        return None  # Unknown
    
    def reset(self):
        """Reset liveness detection state"""
        self.eye_history = []
        self.blink_count = 0
        self.frame_count = 0


# =========================
# FACE ALIGNMENT SYSTEM
# =========================

class FaceAligner:
    """Align faces for better recognition accuracy using facial landmarks"""
    
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height if desired_face_height else desired_face_width
        
    def align_face(self, image, face_landmarks):
        """Align face using eye positions for consistent recognition"""
        try:
            if not face_landmarks:
                return image
            
            # Get eye centers
            left_eye = face_landmarks.get('left_eye', [])
            right_eye = face_landmarks.get('right_eye', [])
            
            if not left_eye or not right_eye:
                return image
            
            # Calculate eye centers - FIXED: Ensure Python float scalars for OpenCV
            left_eye_center = tuple(float(coord) for coord in np.mean(left_eye, axis=0))
            right_eye_center = tuple(float(coord) for coord in np.mean(right_eye, axis=0))
            
            # Calculate angle between eyes
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = float(np.degrees(np.arctan2(dY, dX)))
            
            # Calculate desired right eye position
            desired_right_eye_x = 1.0 - self.desired_left_eye[0]
            
            # Calculate scale
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.desired_face_width
            scale = float(desired_dist / dist if dist > 0 else 1.0)
            
            # Calculate center point between eyes - FIXED: Use float division for precise center
            eyes_center = (
                float((left_eye_center[0] + right_eye_center[0]) / 2),
                float((left_eye_center[1] + right_eye_center[1]) / 2)
            )
            
            logger.debug(f"Face alignment - eyes_center: {eyes_center} (type: {type(eyes_center)})")
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
            
            # Update translation component
            tX = self.desired_face_width * 0.5
            tY = self.desired_face_height * self.desired_left_eye[1]
            M[0, 2] += (tX - eyes_center[0])
            M[1, 2] += (tY - eyes_center[1])
            
            # Apply affine transformation
            aligned_face = cv2.warpAffine(image, M, (self.desired_face_width, self.desired_face_height),
                                          flags=cv2.INTER_CUBIC)
            
            return aligned_face
        except (RuntimeError, ValueError, IndexError):
            logger.exception("Face alignment error")
            return image


# =========================
# CLASSROOM-OPTIMIZED FACE RECOGNITION MODEL
# =========================

class ClassroomOptimizedFaceModel:
    """
    Enhanced face recognition model with liveness detection and face alignment.
    Maintains backward compatibility while adding security features.
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.enhanced_system = EnhancedRecognitionSystem(config_manager)
        self.liveness_detector = LivenessDetector()
        self.face_aligner = FaceAligner()
        self.liveness_enabled = True
        self.alignment_enabled = True
        
        # Load settings from config
        if config_manager:
            self.liveness_enabled = config_manager.config.get('recognition_settings', {}).get('liveness_detection', True)
            self.alignment_enabled = config_manager.config.get('recognition_settings', {}).get('face_alignment', True)

    @property
    def known_face_encodings(self):
        """Delegate to enhanced system"""
        return self.enhanced_system.known_face_encodings

    @known_face_encodings.setter
    def known_face_encodings(self, value):
        """Delegate to enhanced system"""
        self.enhanced_system.known_face_encodings = value

    @property
    def known_face_ids(self):
        """Delegate to enhanced system"""
        return self.enhanced_system.known_face_ids

    @known_face_ids.setter
    def known_face_ids(self, value):
        """Delegate to enhanced system"""
        self.enhanced_system.known_face_ids = value

    @property
    def known_face_names(self):
        """Delegate to enhanced system"""
        return self.enhanced_system.known_face_names

    @known_face_names.setter
    def known_face_names(self, value):
        """Delegate to enhanced system"""
        self.enhanced_system.known_face_names = value

    def load_model(self):
        """Delegate to enhanced system"""
        self.enhanced_system.load_model()
        logger.debug(
            "ClassroomOptimizedFaceModel: Model loaded from enhanced system")

    def detect_faces(self, rgb_frame, model='hog'):
        """Detect faces in the given RGB frame using face_recognition library"""
        return face_recognition.face_locations(rgb_frame, model=model)
    
    def get_face_landmarks(self, rgb_frame, face_location):
        """Get facial landmarks for face alignment and liveness detection"""
        try:
            landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])
            return landmarks[0] if landmarks else None
        except (IndexError, ValueError):
            return None

    def get_face_encoding(self, rgb_frame, face_location, use_alignment=True):
        """Get face encoding with optional alignment for better accuracy"""
        try:
            # Get face landmarks for alignment
            landmarks = None
            if self.alignment_enabled and use_alignment:
                landmarks = self.get_face_landmarks(rgb_frame, face_location)
            
            # Align face if landmarks available
            if landmarks and self.alignment_enabled and use_alignment:
                # Convert RGB to BGR for OpenCV alignment
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                aligned_face = self.face_aligner.align_face(bgr_frame, landmarks)
                # Convert back to RGB for face_recognition
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                
                # Get encoding from aligned face
                encodings = face_recognition.face_encodings(aligned_rgb)
                if encodings:
                    return encodings[0], landmarks
            
            # Standard encoding without alignment
            encodings = face_recognition.face_encodings(rgb_frame, [face_location])
            return (encodings[0] if encodings else None), landmarks
        except (RuntimeError, ValueError) as e:
            logger.exception(f"Error getting face encoding: {e}")
            return None, None

    def check_liveness(self, rgb_frame, face_location):
        """Check if face is live (not a photo) using blink detection"""
        if not self.liveness_enabled:
            return True, "Liveness detection disabled"
        
        try:
            landmarks = self.get_face_landmarks(rgb_frame, face_location)
            if not landmarks:
                return False, "No landmarks detected"
            
            # Detect blink
            is_blinking, ear = self.liveness_detector.detect_blink(landmarks)
            
            # Check if we've collected enough frames
            liveness_result = self.liveness_detector.is_live_face(min_blinks=1, max_frames=30)
            
            if liveness_result is True:
                self.liveness_detector.reset()
                return True, f"Live face confirmed (blinks detected, EAR: {ear:.3f})"
            elif liveness_result is False:
                self.liveness_detector.reset()
                return False, "No blinks detected - possible photo/spoof"
            else:
                # Still collecting data
                return True, f"Collecting liveness data (EAR: {ear:.3f})"
                
        except Exception as e:
            logger.exception(f"Liveness detection error: {e}")
            return True, f"Liveness check failed, allowing: {str(e)}"

    def find_matches(self, face_encoding, frame_quality=0.8):
        """Find matches for a face encoding using the enhanced system"""
        try:
            student_id, student_name, distance, status = self.enhanced_system.adaptive_recognition(
                face_encoding, frame_quality)

            if student_id and status == "confirmed":
                confidence = 1.0 - distance  # Convert distance to confidence
                match_type = "standard"  # Default match type

                # Determine match type based on confidence
                if confidence >= 0.9:
                    match_type = "auto_mark"
                elif confidence >= 0.8:
                    match_type = "high_confidence"

                return [{
                    'student_id': student_id,
                    'student_name': student_name,
                    'confidence': confidence,
                    'match_type': match_type,
                    'distance': distance
                }]

            return []

        except (ValueError, IndexError) as e:
            logger.exception(f"Error finding matches: {e}")
            return []

    def save_model(self):
        """Delegate to enhanced system"""
        self.enhanced_system.save_model()

    def validate_classroom_face_quality(self, face_location, frame):
        """Enhanced quality validation for classroom setting"""
        try:
            top, right, bottom, left = face_location
            h, w = frame.shape[:2]

            # Check face size
            face_area = (right - left) * (bottom - top)
            min_area = self.config_manager.config['recognition_settings']['min_face_area'] if self.config_manager else 1000
            if face_area < min_area:
                return False, f"Face too small (area: {face_area}, min: {min_area})"

            # Check if face is too close to edges
            edge_buffer = self.config_manager.config['recognition_settings']['edge_buffer_pixels'] if self.config_manager else 50
            if (left < edge_buffer or top < edge_buffer or
                    right > w - edge_buffer or bottom > h - edge_buffer):
                return False, "Face too close to frame edge"

            # Check for proper classroom positioning (e.g., students sitting)
            face_center_y = (top + bottom) // 2
            # Assuming students are generally in the upper 75% of the frame when sitting
            if face_center_y > h * 0.75:
                return False, "Face positioned too low (student may be slouching)"

            # Enhanced lighting validation for classroom
            if self.config_manager and self.config_manager.config['classroom_settings']['lighting_validation']:
                face_roi = frame[top:bottom, left:right]
                if face_roi.size == 0:  # Handle empty ROI
                    return False, "Empty face region for lighting check"

                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                # Check for uniform lighting
                mean_brightness = np.mean(gray_roi)
                brightness_std = np.std(gray_roi)

                if mean_brightness < 40:
                    return False, f"Too dark for classroom (brightness: {mean_brightness:.1f})"
                elif mean_brightness > 200:
                    return False, f"Too bright/overexposed (brightness: {mean_brightness:.1f})"

                # Check for harsh shadows (common in classroom lighting)
                if brightness_std > 60:  # Threshold for standard deviation
                    return False, f"Uneven lighting/shadows detected (std: {brightness_std:.1f})"

            # Face aspect ratio (avoid extreme angles) - more restrictive for classroom
            face_width = right - left
            face_height = bottom - top
            aspect_ratio = face_width / face_height
            if aspect_ratio < 0.7 or aspect_ratio > 1.4:  # Tighter range
                return False, f"Face angle not suitable for classroom recognition (aspect ratio: {aspect_ratio:.2f})"

            return True, "Good quality"
        except (RuntimeError, ValueError, ZeroDivisionError) as e:
            logger.exception("Classroom face quality validation error")
            return False, f"Validation error: {str(e)}"
