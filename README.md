# Resume Classification Project

## Overview
This project is a Resume Classification System designed to classify resumes into predefined categories using various machine learning models and ensemble learning techniques. It is implemented in Python and leverages libraries like scikit-learn, XGBoost, and imbalanced-learn for robust and accurate predictions.

### Key Features
- **Text Extraction**: Extracts text from PDF files for preprocessing.
- **Data Preprocessing**: Handles train-test split and oversampling for imbalanced datasets.
- **Model Training**: Supports multiple classifiers, including Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, k-NN, and XGBoost.
- **Ensemble Learning**: Combines predictions from base models using soft voting.
- **Evaluation**: Provides detailed metrics such as accuracy, precision, recall, and F1-score.
- **Customizable**: Allows easy addition of new models and configurations.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/Resume-Classifier.git
   cd Resume-Classifier
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preparation
Split PDFs into training and testing datasets, and extract text:
```bash
python scripts/split_and_extract.py
```

### 2. Handle Class Imbalance
Oversample the training data to address class imbalance:
```bash
python scripts/oversample.py
```

### 3. Train Models
Train individual models:
- Logistic Regression:
  ```bash
  python scripts/training/train_LR_model.py
  ```
- Random Forest:
  ```bash
  python scripts/training/train_Random_model.py
  ```
- SVM:
  ```bash
  python scripts/training/train_SVM_model.py
  ```
- XGBoost:
  ```bash
  python scripts/training/train_XGBoost_model.py
  ```
- Decision Tree:
  ```bash
  python scripts/training/train_DecisionTree_model.py
  ```
- k-NN:
  ```bash
  python scripts/training/train_KNN_model.py
  ```
- Naive Bayes:
  ```bash
  python scripts/training/train_Naive_Bayes_model.py
  ```

### 4. Ensemble Learning
Train and evaluate the ensemble model:
```bash
python scripts/training/train_ensemble_model.py
```

### 5. Calculate Accuracy
Evaluate individual models on the test dataset:
```bash
bash scripts/run_all_accuracy.sh
```

---

## Dependencies

The following dependencies are required and listed in `requirements.txt`:
```
customtkinter==5.2.2
darkdetect==0.8.0
imbalanced-learn==0.12.4
joblib==1.4.2
numpy==2.2.0
pandas==2.2.3
PyPDF2==3.0.1
python-docx==1.1.2
scikit-learn==1.5.2
scipy==1.14.1
xgboost==2.1.3
```

---

## Evaluation
- **Cross-Validation**: Evaluates models using cross-validation during training.
- **Accuracy on Test Data**: Reports final performance on the test set.
- **Detailed Metrics**: Includes precision, recall, F1-score, and confusion matrices.

---

## License
This project is licensed under the MIT License.
