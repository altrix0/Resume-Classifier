import joblib
from tkinter import Toplevel, Listbox, MULTIPLE, Label, Button
MODEL_PATH = "data/models/ensemble_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as file:
        data = joblib.load(file)
    return data["model"], data["vectorizer"], data["label_encoder"]

def classify_resumes(data, model, vectorizer, selected_categories, label_encoder):
    filenames, texts = zip(*data)
    features = vectorizer.transform(texts).toarray()
    predictions = model.predict(features)
    decoded_predictions = label_encoder.inverse_transform(predictions)

    categorized_data = {category: [] for category in selected_categories}
    for filename, label in zip(filenames, decoded_predictions):
        if label in selected_categories:
            categorized_data[label].append(filename)

    print(f"Categorized Data: {categorized_data}")
    return categorized_data

def select_categories(categories):
    selected_categories = []

    def confirm_selection():
        nonlocal selected_categories
        selected_categories = [categories[i] for i in listbox.curselection()]
        selection_window.destroy()

    selection_window = Toplevel()
    selection_window.title("Select Categories")
    Label(selection_window, text="Select categories to classify resumes into:").pack(pady=10)

    listbox = Listbox(selection_window, selectmode=MULTIPLE, width=50, height=15)
    for category in categories:
        listbox.insert("end", category)  # Use "end" as the index
    listbox.pack(pady=10)

    Button(selection_window, text="Confirm", command=confirm_selection).pack(pady=10)
    selection_window.wait_window()
    return selected_categories