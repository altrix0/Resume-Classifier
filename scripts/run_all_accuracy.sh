#!/bin/bash

# Navigate to the directory containing this script
cd "$(dirname "$0")/accuracy_calc"

# List of Python scripts to execute
accuracy_scripts=(
    "decision_tree_acc.py"
    "knn_acc.py"
    "naive_bayes_acc.py"
    "svm_acc.py"
    "LR_acc.py"
    "random_forest_acc.py"
    "xgboost_acc.py"
    "ensemble_acc.py"
)

# Iterate through each script and execute it
for script in "${accuracy_scripts[@]}"; do
    echo "Running $script..."
    python3 "$script"
    if [ $? -ne 0 ]; then
        echo "Error occurred while executing $script. Exiting."
        exit 1
    fi
    echo "$script executed successfully."
done

echo "All accuracy scripts executed."
