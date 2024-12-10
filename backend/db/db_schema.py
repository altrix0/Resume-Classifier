import sqlite3
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists
DB_FILE = os.path.join(DATA_DIR, "database.sqlite")

def initialize_database():
    """
    Initialize the database by creating the necessary tables.
    """
    conn = sqlite3.connect(DB_FILE)
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

    # Create files table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            category TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_database()
    print("Database schema initialized successfully.")
