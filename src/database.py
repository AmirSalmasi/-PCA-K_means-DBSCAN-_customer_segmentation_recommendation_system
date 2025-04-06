import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
from .logging_config import logger

class Database:
    def __init__(self, db_path='data/customer_segmentation.db'):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables."""
        try:
            # Create database directory if it doesn't exist
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create users table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Create model_versions table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    FOREIGN KEY (created_by) REFERENCES users (id)
                )
            ''')
            
            # Create customer_segments table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS customer_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    segment_id INTEGER NOT NULL,
                    model_version_id INTEGER NOT NULL,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_version_id) REFERENCES model_versions (id)
                )
            ''')
            
            # Create audit_log table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            self.conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise

    def create_user(self, username, password, email, role='user'):
        """Create a new user."""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            self.cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', (username, password_hash, email, role))
            self.conn.commit()
            logger.info(f"User {username} created successfully")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Username or email already exists: {username}")
            return False
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}", exc_info=True)
            return False

    def authenticate_user(self, username, password):
        """Authenticate a user."""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            self.cursor.execute('''
                SELECT id, username, role FROM users
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            user = self.cursor.fetchone()
            
            if user:
                # Update last login
                self.cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user[0],))
                self.conn.commit()
                logger.info(f"User {username} authenticated successfully")
                return {'id': user[0], 'username': user[1], 'role': user[2]}
            return None
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}", exc_info=True)
            return None

    def log_audit(self, user_id, action, details=None):
        """Log an audit entry."""
        try:
            self.cursor.execute('''
                INSERT INTO audit_log (user_id, action, details)
                VALUES (?, ?, ?)
            ''', (user_id, action, details))
            self.conn.commit()
            logger.debug(f"Audit log entry created: {action}")
        except Exception as e:
            logger.error(f"Error creating audit log: {str(e)}", exc_info=True)

    def save_model_version(self, version, model_type, performance_metrics, user_id):
        """Save a new model version."""
        try:
            self.cursor.execute('''
                INSERT INTO model_versions (version, model_type, performance_metrics, created_by)
                VALUES (?, ?, ?, ?)
            ''', (version, model_type, performance_metrics, user_id))
            self.conn.commit()
            model_id = self.cursor.lastrowid
            logger.info(f"Model version {version} saved successfully")
            return model_id
        except Exception as e:
            logger.error(f"Error saving model version: {str(e)}", exc_info=True)
            return None

    def save_customer_segments(self, segments_df, model_version_id):
        """Save customer segments to the database."""
        try:
            segments_df['model_version_id'] = model_version_id
            segments_df['created_at'] = datetime.now()
            segments_df.to_sql('customer_segments', self.conn, if_exists='append', index=False)
            logger.info(f"Customer segments saved for model version {model_version_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving customer segments: {str(e)}", exc_info=True)
            return False

    def get_latest_model_version(self, model_type):
        """Get the latest model version."""
        try:
            self.cursor.execute('''
                SELECT * FROM model_versions
                WHERE model_type = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (model_type,))
            return self.cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting latest model version: {str(e)}", exc_info=True)
            return None

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed") 