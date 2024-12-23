import joblib
from tkinter import Toplevel, Listbox, MULTIPLE, Label, Button

# Path to the trained model file
MODEL_PATH = "data/models/ensemble_model.pkl"

def load_model():
    """
    Load the pre-trained machine learning model, vectorizer, and label encoder.

    Returns:
        tuple: Model, vectorizer, and label encoder objects.
    """
    with open(MODEL_PATH, "rb") as file:
        data = joblib.load(file)
    return data["model"], data["vectorizer"], data["label_encoder"]

def classify_resumes(data, model, vectorizer, selected_categories, label_encoder):
    """
    Classify resumes based on the selected categories.

    Args:
        data (list): List of tuples containing filenames and processed texts.
        model: Trained classification model.
        vectorizer: Pre-fitted vectorizer for feature extraction.
        selected_categories (list): Categories selected for classification.
        label_encoder: Label encoder for decoding predictions.

    Returns:
        dict: Resumes categorized by their respective categories.
    """
    filenames, texts = zip(*data)
    features = vectorizer.transform(texts).toarray()
    predictions = model.predict(features)
    decoded_predictions = label_encoder.inverse_transform(predictions)

    categorized_data = {category: [] for category in selected_categories}
    for filename, label in zip(filenames, decoded_predictions):
        if label in selected_categories:
            categorized_data[label].append(filename)

    return categorized_data

def select_categories(categories):
    """
    Display a dialog box to select categories for classification.

    Args:
        categories (list): List of all available categories.

    Returns:
        list: Categories selected by the user.
    """
    selected_categories = []

    def confirm_selection():
        nonlocal selected_categories
        selected_categories = [categories[i] for i in listbox.curselection()]
        selection_window.destroy()

    # Create category selection window
    selection_window = Toplevel()
    selection_window.title("Select Categories")
    Label(selection_window, text="Select categories to classify resumes into:").pack(pady=10)

    # Listbox for category selection
    listbox = Listbox(selection_window, selectmode=MULTIPLE, width=50, height=15)
    for category in categories:
        listbox.insert("end", category)  # Add categories to the listbox
    listbox.pack(pady=10)

    # Confirm button
    Button(selection_window, text="Confirm", command=confirm_selection).pack(pady=10)
    selection_window.wait_window()  # Wait for the user to close the dialog
    return selected_categories
