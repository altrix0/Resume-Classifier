import json

# Load the data from test.json
with open('data/split/test.json', 'r') as f:
    data = json.load(f)

# Extract labels, ensuring uniqueness
labels = set(entry['label'] for entry in data)

# Sort the labels in ascending order
sorted_labels = sorted(labels)

# Save the labels to tests/testlabels.txt
with open('tests/testlabels.txt', 'w') as f:
    for label in sorted_labels:
        f.write(label + '\n')

print("Labels saved to tests/testlabels.txt.")
