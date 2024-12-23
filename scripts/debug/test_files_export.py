import os
import shutil
import json

# Paths
data_raw_dir = 'data/raw'
test_json_path = 'data/split/test.json'
tests_input_dir = 'tests/input'

# Create the destination directory if it doesn't exist
os.makedirs(tests_input_dir, exist_ok=True)

# Load test.json file
with open(test_json_path, 'r') as f:
    test_data = json.load(f)

# Collect the unique labels and their associated files
for entry in test_data:
    label = entry.get("label")
    text = entry.get("text")  # Currently not using text for file copying, but you may use it later
    # Construct the source directory based on the label
    label_dir = os.path.join(data_raw_dir, label)
    
    # Check if the label folder exists
    if os.path.exists(label_dir) and os.path.isdir(label_dir):
        # List all files in the label directory
        files = os.listdir(label_dir)
        
        for file in files:
            src_file = os.path.join(label_dir, file)
            if os.path.isfile(src_file):  # Ensure that the item is a file
                # Define the destination path in the 'tests/input' directory
                dest_file = os.path.join(tests_input_dir, file)
                
                # Copy the file to the new directory
                try:
                    shutil.copy(src_file, dest_file)
                    print(f"Copied {file} from {label_dir} to {tests_input_dir}")
                except Exception as e:
                    print(f"Failed to copy {file}: {e}")
    else:
        print(f"Warning: Directory for label '{label}' does not exist.")
