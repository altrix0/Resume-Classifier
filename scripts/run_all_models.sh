echo "Running Logistic Regression Training..."
python3 scripts/training/train_LR_model.py

echo "Running Random Forest Training..."
python3 scripts/training/train_Random_model.py

echo "Running Naive Bayes Training..."
python3 scripts/training/train_Naive_Bayes_model.py

echo "Running k-NN Training..."
python3 scripts/training/train_KNN_model.py

echo "Running SVM Training..."
python3 scripts/training/train_SVM_model.py

echo "Running Decision Tree Training..."
python3 scripts/training/train_DecisionTree_model.py

echo "Running XGBoost Training..."
python3 scripts/training/train_XGBoost_model.py

echo "Running Ensemble Training..."
python3 scripts/training/train_ensemble_model.py

echo "Training complete!"
