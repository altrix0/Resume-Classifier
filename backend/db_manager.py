import sqlite3
from datetime import datetime
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "database.sqlite")

# Database initialization
def initialize_database():
    """
    Create or connect to the SQLite database and initialize tables.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            output_path TEXT NOT NULL
        )
    """)

    # Create models table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE NOT NULL,
            model_data BLOB NOT NULL
        )
    """)

    # Create logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            log_type TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)

    # Create files table for tracking uploaded files
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,
            category TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)

    conn.commit()
    conn.close()

# CRUD operations

# Insert a new session
def insert_session(session_name, output_path):
    """
    Insert a new sorting session into the database.
    Ensures session names are unique.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO sessions (session_name, created_at, output_path)
            VALUES (?, ?, ?)
        """, (session_name, created_at, output_path))
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Session name '{session_name}' already exists.")
    finally:
        conn.close()

# Log messages (progress or errors)
def log_message(session_id, log_type, message):
    """
    Log a progress or error message for a specific session.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO logs (session_id, log_type, message, created_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, log_type, message, created_at))
    conn.commit()
    conn.close()

# Insert file metadata
def insert_file(session_id, file_name, file_path, status="pending", category=None):
    """
    Insert a record for an uploaded file linked to a specific session.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO files (session_id, file_name, file_path, status, category)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, file_name, file_path, status, category))
    conn.commit()
    conn.close()

# Retrieve session details
def retrieve_session(session_name):
    """
    Retrieve a session by its name.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM sessions WHERE session_name = ?
    """, (session_name,))
    session = cursor.fetchone()
    conn.close()
    if session:
        return session
    else:
        raise ValueError(f"Session '{session_name}' not found.")

# Retrieve logs for a specific session
def get_logs(session_id):
    """
    Retrieve logs for a specific session.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT log_type, message, created_at FROM logs WHERE session_id = ?
    """, (session_id,))
    logs = cursor.fetchall()
    conn.close()
    return logs

# Retrieve files linked to a specific session
def get_files(session_id):
    """
    Retrieve files for a specific session.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT file_name, status, category FROM files WHERE session_id = ?
    """, (session_id,))
    files = cursor.fetchall()
    conn.close()
    return files

# Initialize the database when this script is run
if __name__ == "__main__":
    initialize_database()
    print("Database initialized.")
