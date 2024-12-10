from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def classify_resumes(data, model, vectorizer, selected_categories):
    filenames, texts = zip(*data)
    features = vectorizer.transform(texts).toarray()  # Convert to dense matrix
    predictions = model.predict(features)
    categorized_data = {label: [] for label in selected_categories}
    for filename, label in zip(filenames, predictions):
        if label in selected_categories:  # Only include selected categories
            categorized_data[label].append(filename)
    return categorized_data
