import pickle

models_dir = "data/models"
model_files = [
    "logistic_regression_balanced.pkl",
    "random_forest.pkl",
    "naive_bayes.pkl",
    "knn.pkl",
    "svm.pkl",
    "decision_tree.pkl",
    "xgboost.pkl",
]

for model_file in model_files:
    try:
        with open(f"{models_dir}/{model_file}", "rb") as file:
            data = pickle.load(file)
            print(f"{model_file}: Loaded successfully. Keys: {data.keys()}")
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
