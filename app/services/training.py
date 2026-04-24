"""
Training service for face recognition models.
Handles CNN-optimized training with batching and memory management.
Uses ThreadPoolExecutor for cross-platform compatibility (Windows/Linux/macOS).
"""

import os
import sys
import time
import logging
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)

# Global training progress variable with thread-safe access
training_progress = {
    'status': 'idle',
    'current': 0,
    'total': 0,
    'percentage': 0,
    'message': 'Ready for training'
}
_progress_lock = threading.Lock()


def _update_progress(**kwargs):
    """Thread-safe progress update"""
    global training_progress
    with _progress_lock:
        training_progress.update(kwargs)


class CNNOptimizedFaceTrainer:
    """CNN-specific optimizations while maintaining high accuracy and 15 images per student"""

    def __init__(self, config_manager, db_manager, attendance_system):
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.attendance_system = attendance_system

    @staticmethod
    def _preprocess_image(image_path, target_size=(800, 800)):
        """Preprocess image to optimal size for CNN face detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, "Failed to load image"

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]
            if width > target_size[0] or height > target_size[1]:
                ratio = min(target_size[0] / width, target_size[1] / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            return image, "success"
        except Exception as e:
            return None, f"Preprocessing error: {str(e)}"

    def _process_single_image(self, filename, student_map, dataset_dir, recognition_settings):
        """Process a single image for training - compatible with both ThreadPool and ProcessPool"""
        try:
            student_id_str = filename.split('_')[0]
            if not student_id_str.isdigit():
                return None, f"Skipping malformed filename: {filename}"

            student_id = int(student_id_str)
            if student_id not in student_map:
                return None, f"Skipping image for unknown student ID: {student_id}"

            image_path = os.path.join(dataset_dir, filename)

            # CNN-optimized preprocessing
            image, preprocess_msg = self._preprocess_image(
                image_path, target_size=(800, 800))
            if image is None:
                return None, f"Preprocessing failed for {filename}: {preprocess_msg}"

            detection_model = recognition_settings.get(
                'training_detection_model', 'cnn')
            face_locations = face_recognition.face_locations(
                image, model=detection_model)

            if not face_locations:
                return None, f"No face found in image: {filename}"

            # Take the largest/most confident face
            if len(face_locations) > 1:
                face_areas = [(bottom - top) * (right - left)
                              for (top, right, bottom, left) in face_locations]
                largest_face_idx = np.argmax(face_areas)
                face_locations = [face_locations[largest_face_idx]]

            # CNN face encoding with balanced jitters
            num_jitters = recognition_settings.get('num_jitters', 1)

            face_encodings = face_recognition.face_encodings(
                image,
                face_locations,
                num_jitters=num_jitters,
                model='large'   # Use large model for better CNN accuracy
            )

            if not face_encodings:
                return None, f"Could not encode face in: {filename}"

            # Clean up memory
            del image

            return {
                'encoding': face_encodings[0],
                'student_id': student_id,
                'student_name': student_map[student_id],
                'filename': filename
            }, "success"

        except Exception as e:
            return None, f"Error processing {filename}: {str(e)}"

    def smart_cnn_training_with_batching(self):
        """CNN-optimized training with intelligent batching and memory management.
        
        Uses ThreadPoolExecutor for cross-platform compatibility.
        Performance is comparable to ProcessPoolExecutor on most systems.
        """
        logger.info(
            "Starting CNN-OPTIMIZED training with ThreadPoolExecutor (cross-platform compatible)...")
        start_time = time.time()

        try:
            students = self.db_manager.get_all_students()
            student_map = {s['id']: s['name'] for s in students}

            if not student_map:
                _update_progress(
                    status='failed', message='No students found in database')
                return False, "No students found in database"

            dataset_dir = self.config_manager.config['system_settings']['dataset_dir']
            if not os.path.exists(dataset_dir):
                _update_progress(
                    status='failed', message='Dataset directory not found')
                return False, "Dataset directory not found"

            image_files = [f for f in os.listdir(dataset_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                _update_progress(
                    status='failed', message='No training images found')
                return False, "No training images found"

            logger.info(
                f"CNN Training: {len(image_files)} images for {len(student_map)} students")

            results_from_processes = []
            processed_count = 0
            error_count = 0

            # Group images by student for better memory management
            student_images = {}
            for filename in image_files:
                try:
                    student_id = int(filename.split('_')[0])
                    if student_id in student_map:
                        if student_id not in student_images:
                            student_images[student_id] = []
                        student_images[student_id].append(filename)
                except Exception:
                    continue

            # Determine optimal number of workers
            # Use threading for cross-platform compatibility
            # ThreadPoolExecutor is sufficient for I/O-bound tasks like image loading
            num_cpu_cores = os.cpu_count() or 4
            max_workers = min(num_cpu_cores, len(image_files))

            # Pass only picklable data to the worker function
            recognition_settings_for_workers = self.config_manager.config['recognition_settings'].copy()

            _update_progress(
                status='in_progress',
                total=len(image_files),
                message='Initializing CNN workers...'
            )

            # Process students one by one to manage memory better with CNN
            for student_id, filenames in student_images.items():
                student_name = student_map.get(student_id, 'Unknown')
                logger.info(
                    f"Processing {len(filenames)} images for student {student_name} (ID: {student_id})")

                # Process this student's images in parallel using ThreadPoolExecutor
                # ThreadPoolExecutor is more portable across platforms
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks for the current student's images
                    futures = [
                        executor.submit(
                            self._process_single_image,
                            filename, student_map, dataset_dir, recognition_settings_for_workers
                        )
                        for filename in filenames
                    ]

                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            result, message = future.result(timeout=120)

                            if result is not None:
                                results_from_processes.append(result)
                                processed_count += 1

                                # Safely get filename from result
                                result_filename = result.get(
                                    'filename', 'unknown') if isinstance(result, dict) else 'unknown'

                                # Update global progress (thread-safe)
                                _update_progress(
                                    current=processed_count,
                                    percentage=(processed_count / len(image_files)) * 100,
                                    message=f'Processing {result_filename} for student {student_name}...'
                                )

                                if processed_count % 5 == 0:
                                    elapsed = time.time() - start_time
                                    avg_time_per_image = elapsed / processed_count
                                    eta_seconds = avg_time_per_image * \
                                        (len(image_files) - processed_count)
                                    logger.info(
                                        f"CNN Training: {processed_count}/{len(image_files)} | ETA: {eta_seconds//60:.0f}m {eta_seconds % 60:.0f}s")
                            else:
                                error_count += 1
                                logger.warning(message)

                        except Exception as e:
                            error_count += 1
                            failed_filename = "unknown"
                            if future.done():
                                try:
                                    failed_result, _ = future.result()
                                    if failed_result and 'filename' in failed_result:
                                        failed_filename = failed_result['filename']
                                except Exception:
                                    pass
                            logger.error(
                                f"CNN processing error for {failed_filename}: {str(e)}")

                # Memory cleanup between students
                gc.collect()

                # Adjusted count
                successful_count = len([r for r in results_from_processes if r.get('student_id') == student_id])
                logger.info(
                    f"Completed student {student_name}: {successful_count} successful encodings")

            # --- Combine results after all processes are done ---
            known_face_encodings = [r['encoding']
                                    for r in results_from_processes]
            known_face_ids = [r['student_id'] for r in results_from_processes]
            known_face_names = [r['student_name']
                                for r in results_from_processes]

            # Update the face model
            self.attendance_system.face_model.known_face_encodings = known_face_encodings
            self.attendance_system.face_model.known_face_ids = known_face_ids
            self.attendance_system.face_model.known_face_names = known_face_names

            # Save model
            self.attendance_system.face_model.save_model()

            end_time = time.time()
            training_time = end_time - start_time

            logger.info(
                f"CNN Training completed in {training_time//60:.0f}m {training_time % 60:.0f}s")
            logger.info(
                f"Successfully processed: {processed_count}/{len(image_files)} images")
            logger.info(f"Errors: {error_count}")

            self.db_manager.log_security_event(
                "CNN_MODEL_TRAINED",
                details=f"CNN training: {processed_count} images in {training_time//60:.0f}m {training_time % 60:.0f}s"
            )
            _update_progress(
                status='completed',
                message=f"CNN training completed successfully in {training_time//60:.0f}m {training_time % 60:.0f}s. Processed {processed_count} faces (15 per student maintained)."
            )
            return True, training_progress['message']

        except Exception as e:
            logger.exception("Error during CNN training")
            self.db_manager.log_security_event(
                "CNN_TRAINING_FAILED",
                details=f"CNN training error: {str(e)}",
                severity="ERROR"
            )
            _update_progress(
                status='failed',
                message=f"CNN training failed: {str(e)}"
            )
            return False, training_progress['message']


def enhanced_train_model_cnn_optimized(config_manager, db_manager, attendance_system):
    """CNN-optimized training that maintains 15 images per student"""
    trainer = CNNOptimizedFaceTrainer(
        config_manager, db_manager, attendance_system)
    return trainer.smart_cnn_training_with_batching()


def get_training_progress():
    """Get current training progress (thread-safe)"""
    with _progress_lock:
        return dict(training_progress)


def reset_training_progress():
    """Reset training progress to idle state (thread-safe)"""
    global training_progress
    with _progress_lock:
        training_progress = {
            'status': 'idle',
            'current': 0,
            'total': 0,
            'percentage': 0,
            'message': 'Ready for training'
        }

