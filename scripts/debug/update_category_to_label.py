# import json

# # Paths
# train_balanced_path = "data/split/train_balanced.json"
# updated_train_balanced_path = "data/split/train_balanced_updated.json"

# # Update key
# with open(train_balanced_path, "r") as file:
#     data = json.load(file)

# # Change 'category' to 'label'
# updated_data = [{"text": item["text"], "label": item["category"]} for item in data]

# # Save updated data
# with open(updated_train_balanced_path, "w") as file:
#     json.dump(updated_data, file, indent=4)

# print(f"Updated 'train_balanced.json' saved to '{updated_train_balanced_path}'")


import json

def convert_to_label(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for item in data:
        item['label'] = item.pop('category')

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

convert_to_label('data/split/test.json', 'data/split/test_updated.json')
