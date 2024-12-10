import db_manager

# Test insert session
def test_insert_session():
    try:
        db_manager.insert_session("Test Session", "/path/to/output")
        print("Session inserted successfully.")
    except ValueError as e:
        print(f"Error: {e}")

# Test log message
def test_log_message():
    try:
        db_manager.log_message(1, "progress", "Started file processing.")
        db_manager.log_message(1, "error", "File not found.")
        print("Log messages inserted successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Test file insertion
def test_insert_file():
    try:
        db_manager.insert_file(1, "resume1.pdf", "/path/to/resume1.pdf", status="pending", category="Engineering")
        db_manager.insert_file(1, "resume2.pdf", "/path/to/resume2.pdf", status="processed", category="Sales")
        print("Files inserted successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Test retrieve session
def test_retrieve_session():
    try:
        session = db_manager.retrieve_session("Test Session")
        print(f"Session retrieved: {session}")
    except ValueError as e:
        print(f"Error: {e}")

# Test retrieving logs
def test_get_logs():
    try:
        logs = db_manager.get_logs(1)
        print(f"Logs: {logs}")
    except Exception as e:
        print(f"Error: {e}")

# Test retrieving files
def test_get_files():
    try:
        files = db_manager.get_files(1)
        print(f"Files: {files}")
    except Exception as e:
        print(f"Error: {e}")

# Run all tests
if __name__ == "__main__":
    test_insert_session()
    test_log_message()
    test_insert_file()
    test_retrieve_session()
    test_get_logs()
    test_get_files()
