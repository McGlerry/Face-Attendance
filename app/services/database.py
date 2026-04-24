# app/services/database.py
import os
import sqlite3
import time
import threading
import queue as queue_module
import bcrypt
from contextlib import contextmanager
from datetime import datetime, timedelta
import logging

from ..utils.config import DB_FILE, DATASET_DIR


logger = logging.getLogger(__name__)

# =========================
# OPTIMIZED DATABASE MANAGER (With connection pooling, caching, indexes, and optimized queries)
# =========================


class OptimizedDBManager:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        # Store database file path for reference
        self.db_file = DB_FILE
        # Connection pool with 5 connections
        self.connection_pool = queue_module.Queue(maxsize=5)
        for _ in range(5):
            # Allow multi-thread access
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self.connection_pool.put(conn)

        # Caching for students and classes with TTL
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes TTL

        self.setup_tables()
        self.create_indexes()

    @contextmanager
    def cursor(self):
        conn = self.connection_pool.get()
        stale_connection = False
        try:
            # Validate connection is still alive
            conn.execute("SELECT 1")
        except Exception:
            # Connection is stale, create a new one
            stale_connection = True
            try:
                conn.close()
            except Exception:
                pass
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            conn.row_factory = sqlite3.Row
        try:
            yield conn.cursor()
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            logger.exception(f"Database operation failed: {e}")
            raise RuntimeError(f"Database operation failed: {e}") from e
        finally:
            # Only return valid connections to pool
            if not stale_connection:
                self.connection_pool.put(conn)
            else:
                # Stale connection was replaced, don't return it
                pass

    def create_indexes(self):
        """Create database indexes for performance optimization"""
        with self.cursor() as c:
            # Index on attendance for faster queries
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance (student_id, date)')
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_attendance_class_date ON attendance (class_id, date)')
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance (timestamp)')

            # Index on students for faster lookups
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_students_name ON students (name)')

            # Index on classes for active class queries
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_classes_code ON classes (class_code)')

            # Index on security logs
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_security_logs_timestamp ON security_logs (timestamp)')
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_security_logs_event_type ON security_logs (event_type)')

            logger.info("Database indexes created/verified")

    def get_cached(self, key):
        """Get item from cache if not expired"""
        if key in self.cache:
            item, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return item
            else:
                del self.cache[key]
        return None

    def set_cached(self, key, value):
        """Set item in cache with current timestamp"""
        self.cache[key] = (value, time.time())

    def setup_tables(self):
        with self.cursor() as c:
            # Users table for admin authentication
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT 'admin',
                    is_active INTEGER DEFAULT 1,
                    last_login TEXT,
                    created_date TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Students table
            c.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    student_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    created_date TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Classes table
            c.execute('''
                CREATE TABLE IF NOT EXISTS classes (
                    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name TEXT NOT NULL,
                    class_code TEXT UNIQUE,
                    start_time TEXT,
                    end_time TEXT,
                    days_of_week TEXT,
                    created_date TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Class enrollment table
            c.execute('''
                CREATE TABLE IF NOT EXISTS class_enrollment (
                    class_id INTEGER,
                    student_id INTEGER,
                    enrollment_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                    PRIMARY KEY (class_id, student_id)
                )
            ''')

            # Enhanced attendance table with confidence and match type
            c.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_id INTEGER,
                    student_id INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    date TEXT,
                    confidence_score REAL,
                    match_type TEXT DEFAULT 'standard',
                    status TEXT DEFAULT 'present',
                    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                    UNIQUE(class_id, student_id, date)
                )
            ''')

            # NEW: Security logs table (simplified, no hashing/encryption)
            c.execute('''
                CREATE TABLE IF NOT EXISTS security_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    student_id INTEGER,
                    confidence_score REAL,
                    details TEXT,
                    class_id INTEGER,
                    severity TEXT DEFAULT 'INFO'
                )
            ''')

    def log_security_event(self, event_type, student_id=None, confidence_score=None, details="", class_id=None, severity="INFO"):
        """Log security events for audit trail"""
        if self.config_manager and self.config_manager.config['system_settings']['security_logging']:
            try:
                with self.cursor() as c:
                    c.execute("""
                        INSERT INTO security_logs (event_type, student_id, confidence_score, details, class_id, severity)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (event_type, student_id, confidence_score, details, class_id, severity))
            except Exception as e:
                logger.exception(f"Failed to log security event: {e}")

    def add_student(self, student_id, name, email="", phone=""):
        try:
            with self.cursor() as c:
                c.execute("INSERT INTO students (student_id, name, email, phone) VALUES (?, ?, ?, ?)",
                          (student_id, name, email, phone))
            self.log_security_event(
                "STUDENT_ADDED", student_id, details=f"Name: {name}")
            return True, "Student added successfully."
        except sqlite3.IntegrityError:
            return False, f"Student with ID {student_id} already exists."
        except Exception as e:
            logger.exception("add_student error")
            return False, str(e)

    def get_all_students(self):
        with self.cursor() as c:
            c.execute(
                "SELECT student_id, name, email, phone FROM students ORDER BY name")
            return [{"id": row[0], "name": row[1], "email": row[2], "phone": row[3]}
                    for row in c.fetchall()]

    def get_student_by_id(self, student_id):
        with self.cursor() as c:
            c.execute(
                "SELECT student_id, name, email, phone FROM students WHERE student_id = ?", (student_id,))
            row = c.fetchone()
            if row:
                return {"id": row[0], "name": row[1], "email": row[2], "phone": row[3]}
            return None

    def update_student(self, student_id, new_name, email="", phone=""):
        try:
            with self.cursor() as c:
                c.execute("UPDATE students SET name = ?, email = ?, phone = ? WHERE student_id = ?",
                          (new_name, email, phone, student_id))
            return True, "Student updated successfully."
        except Exception as e:
            logger.exception("update_student error")
            return False, str(e)

    def delete_student(self, student_id):
        try:
            with self.cursor() as c:
                # ON DELETE CASCADE should handle attendance and enrollment, but explicit is safer
                c.execute(
                    "DELETE FROM attendance WHERE student_id = ?", (student_id,))
                c.execute(
                    "DELETE FROM class_enrollment WHERE student_id = ?", (student_id,))
                c.execute(
                    "DELETE FROM students WHERE student_id = ?", (student_id,))

            # Delete face images
            for f in os.listdir(DATASET_DIR):
                if f.startswith(str(student_id) + "_"):
                    try:
                        os.remove(os.path.join(DATASET_DIR, f))
                    except Exception:
                        pass

            self.log_security_event("STUDENT_DELETED", student_id)
            return True, "Student and all associated data deleted successfully."
        except Exception as e:
            logger.exception("delete_student error")
            return False, str(e)

    def add_class(self, class_name, class_code, start_time, end_time, days_of_week=""):
        try:
            with self.cursor() as c:
                c.execute(
                    "INSERT INTO classes (class_name, class_code, start_time, end_time, days_of_week) VALUES (?, ?, ?, ?, ?)",
                    (class_name, class_code, start_time, end_time, days_of_week))
            return True, "Class added successfully."
        except sqlite3.IntegrityError:
            return False, "Class code already exists."
        except Exception as e:
            logger.exception("add_class error")
            return False, str(e)

    def get_all_classes(self):
        with self.cursor() as c:
            c.execute("SELECT * FROM classes ORDER BY class_name")
            return [{"class_id": r[0], "class_name": r[1], "class_code": r[2],
                     "start_time": r[3], "end_time": r[4], "days_of_week": r[5]}
                    for r in c.fetchall()]

    def mark_attendance(self, student_id, class_id, confidence_score=0.0, match_type="standard", status="present"):
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")

            # Use a more robust check for existing attendance
            with self.cursor() as c:
                c.execute("""
                    SELECT attendance_id, timestamp FROM attendance
                    WHERE class_id = ? AND student_id = ? AND date = ?
                """, (class_id, student_id, date_str))

                existing = c.fetchone()
                if existing:
                    logger.info(
                        f"Attendance already exists for student {student_id} in class {class_id} on {date_str}")
                    return True, f"Already marked for today at {existing['timestamp']}"

            # Mark new attendance with explicit transaction
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self.cursor() as c:
                c.execute("BEGIN IMMEDIATE")  # Prevent concurrent writes
                try:
                    # Double-check after acquiring lock
                    c.execute("""
                        SELECT attendance_id FROM attendance
                        WHERE class_id = ? AND student_id = ? AND date = ?
                    """, (class_id, student_id, date_str))

                    if c.fetchone():
                        c.execute("ROLLBACK")
                        return True, "Already marked (race condition avoided)"

                    c.execute("""
                        INSERT INTO attendance
                        (class_id, student_id, timestamp, date, confidence_score, match_type, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (class_id, student_id, timestamp, date_str, confidence_score, match_type, status))

                    c.execute("COMMIT")

                except Exception as e:
                    c.execute("ROLLBACK")
                    raise e

            # Log attendance event
            self.log_security_event("ATTENDANCE_MARKED", student_id, confidence_score,
                                    f"Match type: {match_type}, Class: {class_id}, Status: {status}", class_id)
            logger.info(
                f"NEW attendance marked for student {student_id} in class {class_id}")
            return True, "Attendance marked successfully."

        except sqlite3.IntegrityError:
            # This should now be rare due to our double-checking
            return True, "Already marked (integrity constraint)"
        except Exception as e:
            logger.exception("mark_attendance error")
            return False, str(e)

    def get_today_attendance_count(self):
        """Get count of only the attendance records that were actually marked for today"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with self.cursor() as c:
                c.execute("""
                    SELECT COUNT(*) as count FROM attendance
                    WHERE date = ?
                """, (today,))
                result = c.fetchone()
                return result['count'] if result else 0
        except Exception as e:
            logger.exception("get_today_attendance_count error")
            return 0

    def get_attendance_records(self, start_date, end_date, class_id=None):
        """Enhanced method to retrieve attendance records including absent students"""
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Generate list of dates
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            dates.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += timedelta(days=1)

        with self.cursor() as c:
            # Get classes to include
            if class_id and str(class_id).lower() != 'all':
                c.execute(
                    "SELECT class_id, class_name, start_time FROM classes WHERE class_id = ?", (int(class_id),))
                classes = c.fetchall()
            else:
                c.execute(
                    "SELECT class_id, class_name, start_time FROM classes ORDER BY start_time")
                classes = c.fetchall()

            all_records = []
            debug_info = {
                'classes_found': len(classes),
                'dates': dates
            }

            if len(classes) == 0:
                logger.warning("No classes found in database for attendance report")
                logger.warning(f"Debug info: {debug_info}")
                return all_records

            for cls in classes:
                class_id_val = cls['class_id']
                class_name = cls['class_name']
                start_time = cls['start_time'] or '00:00'  # Default if None

                # Get enrolled students for this class
                c.execute("""
                    SELECT s.student_id, s.name
                    FROM students s
                    JOIN class_enrollment ce ON s.student_id = ce.student_id
                    WHERE ce.class_id = ?
                    ORDER BY s.name
                """, (class_id_val,))
                enrolled_students = c.fetchall()

                debug_info[f'class_{class_id_val}_students'] = len(enrolled_students)

                if len(enrolled_students) == 0:
                    logger.info(f"No enrolled students found for class {class_name} (ID: {class_id_val})")
                    continue

                for student in enrolled_students:
                    student_id = student['student_id']
                    student_name = student['name']

                    for date_str in dates:
                        # Check if attendance exists for this student, class, date
                        c.execute("""
                            SELECT a.timestamp, a.confidence_score, a.status, a.match_type
                            FROM attendance a
                            WHERE a.class_id = ? AND a.student_id = ? AND a.date = ?
                        """, (class_id_val, student_id, date_str))
                        attendance = c.fetchone()

                        if attendance:
                            # Present or late
                            all_records.append({
                                'student_id': student_id,
                                'name': student_name,
                                'class_name': class_name,
                                'timestamp': attendance['timestamp'],
                                'confidence_score': attendance['confidence_score'],
                                'status': attendance['status'] or 'present',
                                'match_type': attendance['match_type'] or 'standard',
                                'date': date_str,
                                'class_id': class_id_val,
                                'start_time': start_time
                            })
                        else:
                            # Absent
                            all_records.append({
                                'student_id': student_id,
                                'name': student_name,
                                'class_name': class_name,
                                'timestamp': None,
                                'confidence_score': 0.0,
                                'status': 'absent',
                                'match_type': 'absent',
                                'date': date_str,
                                'class_id': class_id_val,
                                'start_time': start_time
                            })

            # Sort by class start_time, then by student name, then by date
            all_records.sort(key=lambda x: (
                x['start_time'], x['name'], x['date']))

            logger.info(
                f"Generated {len(all_records)} attendance records (including absent)")
            logger.info(f"Debug info: {debug_info}")
            return all_records

    def is_student_enrolled(self, student_id, class_id):
        with self.cursor() as c:
            c.execute("SELECT 1 FROM class_enrollment WHERE student_id = ? AND class_id = ?",
                      (student_id, class_id))
            return c.fetchone() is not None

    def get_class_by_id(self, class_id):
        with self.cursor() as c:
            c.execute(
                "SELECT class_id, class_name, class_code, start_time, end_time, days_of_week FROM classes WHERE class_id = ?", (class_id,))
            row = c.fetchone()
            if row:
                return {"class_id": row[0], "class_name": row[1], "class_code": row[2],
                        "start_time": row[3], "end_time": row[4], "days_of_week": row[5]}
            return None

    def get_enrolled_students(self, class_id):
        with self.cursor() as c:
            c.execute("""
                SELECT s.student_id, s.name, s.email, s.phone
                FROM students s
                JOIN class_enrollment ce ON s.student_id = ce.student_id
                WHERE ce.class_id = ?
                ORDER BY s.name
            """, (class_id,))
            return [{"id": row[0], "name": row[1], "email": row[2], "phone": row[3]}
                    for row in c.fetchall()]

    def get_unenrolled_students(self, class_id):
        with self.cursor() as c:
            c.execute("""
                SELECT s.student_id, s.name, s.email, s.phone
                FROM students s
                LEFT JOIN class_enrollment ce ON s.student_id = ce.student_id AND ce.class_id = ?
                WHERE ce.class_id IS NULL
                ORDER BY s.name
            """, (class_id,))
            return [{"id": row[0], "name": row[1], "email": row[2], "phone": row[3]}
                    for row in c.fetchall()]

    def enroll_student(self, student_id, class_id):
        try:
            with self.cursor() as c:
                c.execute("INSERT INTO class_enrollment (class_id, student_id) VALUES (?, ?)",
                          (class_id, student_id))
            return True, "Student enrolled successfully."
        except sqlite3.IntegrityError:
            return False, "Student is already enrolled in this class."
        except Exception as e:
            logger.exception("enroll_student error")
            return False, str(e)

    def unenroll_student(self, student_id, class_id):
        try:
            with self.cursor() as c:
                c.execute("DELETE FROM class_enrollment WHERE class_id = ? AND student_id = ?",
                          (class_id, student_id))
            return True, "Student unenrolled successfully."
        except Exception as e:
            logger.exception("unenroll_student error")
            return False, str(e)

    def get_security_logs(self, limit=100):
        """Get recent security events for admin review"""
        with self.cursor() as c:
            c.execute("""
                SELECT sl.timestamp, sl.event_type, sl.student_id, sl.class_id, 
                       s.name, sl.confidence_score, sl.details, c.class_name, sl.severity
                FROM security_logs sl
                LEFT JOIN students s ON sl.student_id = s.student_id
                LEFT JOIN classes c ON sl.class_id = c.class_id
                ORDER BY sl.timestamp DESC
                LIMIT ?
            """, (limit,))
            return c.fetchall()

    # =========================
    # USER AUTHENTICATION METHODS
    # =========================

    def create_default_admin(self):
        """Create default admin user if not exists"""
        try:
            with self.cursor() as c:
                c.execute("SELECT COUNT(*) FROM users")
                if c.fetchone()[0] == 0:
                    # Create default admin - use environment variable or fallback
                    default_password = os.environ.get('FLASK_ADMIN_PASSWORD', 'admin123')
                    # Use bcrypt for secure password hashing
                    password_hash = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt(rounds=12))
                    c.execute("""
                        INSERT INTO users (username, password_hash, email, role)
                        VALUES (?, ?, ?, ?)
                    """, ('admin', password_hash, 'admin@system.com', 'admin'))
                    logger.info("Default admin user created")
                    return True
        except Exception as e:
            logger.exception("Error creating default admin")
            return False


    def verify_user(self, username, password):
        """Verify user credentials using bcrypt"""
        try:
            with self.cursor() as c:
                c.execute("""
                    SELECT user_id, username, password_hash, role, is_active
                    FROM users
                    WHERE username = ?
                """, (username,))
                user = c.fetchone()
                if user:
                    # Verify password with bcrypt
                    stored_hash = user['password_hash']
                    if isinstance(stored_hash, str):
                        stored_hash = stored_hash.encode('utf-8')
                    if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                        # Update last login
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        c.execute("UPDATE users SET last_login = ? WHERE user_id = ?",
                                  (timestamp, user['user_id']))
                        return {
                            'user_id': user['user_id'],
                            'username': user['username'],
                            'role': user['role'],
                            'is_active': bool(user['is_active'])
                        }
                return None
        except Exception as e:
            logger.exception("Error verifying user")
            return None


    def get_all_users(self):
        """Get all users"""
        try:
            with self.cursor() as c:
                c.execute("""
                    SELECT user_id, username, email, role, is_active, last_login, created_date
                    FROM users
                    ORDER BY created_date DESC
                """)
                users = []
                for row in c.fetchall():
                    users.append({
                        'user_id': row['user_id'],
                        'username': row['username'],
                        'email': row['email'],
                        'role': row['role'],
                        'is_active': bool(row['is_active']),
                        'last_login': row['last_login'],
                        'created_date': row['created_date']
                    })
                return users
        except Exception as e:
            logger.exception("Error getting users")
            return []

    def add_user(self, username, password, email="", role="admin"):
        """Add new user with bcrypt password hashing"""
        try:
            # Use bcrypt for secure password hashing
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))
            with self.cursor() as c:
                c.execute("""
                    INSERT INTO users (username, password_hash, email, role)
                    VALUES (?, ?, ?, ?)
                """, (username, password_hash, email, role))
                return True, f"User '{username}' created successfully"
        except sqlite3.IntegrityError:
            return False, f"Username '{username}' already exists"
        except Exception as e:
            logger.exception("Error adding user")
            return False, str(e)


    def update_user_password(self, user_id, new_password):
        """Update user password with bcrypt hashing"""
        try:
            # Use bcrypt for secure password hashing
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt(rounds=12))
            with self.cursor() as c:
                c.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                          (password_hash, user_id))
                return True, "Password updated successfully"
        except Exception as e:
            logger.exception("Error updating password")
            return False, str(e)


    def delete_user(self, user_id):
        """Delete user"""
        try:
            with self.cursor() as c:
                c.execute("DELETE FROM users WHERE user_id = ? AND role != 'admin'",
                          (user_id,))
                return True, "User deleted successfully"
        except Exception as e:
            logger.exception("Error deleting user")
            return False, str(e)

    def get_user_by_username(self, username):
        """Get user by username"""
        try:
            with self.cursor() as c:
                c.execute("""
                    SELECT user_id, username, email, role, is_active, last_login, created_date
                    FROM users
                    WHERE username = ?
                """, (username,))
                row = c.fetchone()
                if row:
                    return {
                        'user_id': row['user_id'],
                        'username': row['username'],
                        'email': row['email'],
                        'role': row['role'],
                        'is_active': bool(row['is_active']),
                        'last_login': row['last_login'],
                        'created_date': row['created_date']
                    }
                return None
        except Exception as e:
            logger.exception("Error getting user by username")
            return None

    def update_last_login(self, username):
        """Update last login timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self.cursor() as c:
                c.execute("UPDATE users SET last_login = ? WHERE username = ?",
                          (timestamp, username))
                return True
        except Exception as e:
            logger.exception("Error updating last login")
            return False
