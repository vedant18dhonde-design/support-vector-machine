# Support Vector Machine — Breast Cancer Classification

A binary classification project using Support Vector Machines (SVM) to predict whether a breast tumor is malignant or benign, based on the Wisconsin Breast Cancer dataset.

## Overview

This project builds and tunes an SVM classifier using scikit-learn's built-in `load_breast_cancer` dataset. It covers the full workflow: data loading, train/test split, baseline model training, evaluation, and hyperparameter tuning via `GridSearchCV`.

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numeric features computed from digitized images of breast mass cell nuclei (e.g. radius, texture, perimeter, area, smoothness)
- **Target:** Binary — malignant (0) or benign (1)

## Approach

1. Load the dataset and convert it into a pandas DataFrame.
2. Split data into training (67%) and test (33%) sets.
3. Train a baseline `SVC()` model (default RBF kernel).
4. Evaluate using a confusion matrix and classification report.
5. Tune hyperparameters `C` and `gamma` using `GridSearchCV`.
6. Re-evaluate the tuned model on the test set.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (`SVC`, `GridSearchCV`, `train_test_split`)
- matplotlib, seaborn (for visualization support)

## Hyperparameter Grid

```python
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}
```

`GridSearchCV` searches this grid with 5-fold cross-validation (default) to find the `C`/`gamma` combination that maximizes classification performance.

## Results

The tuned model's best parameters and test-set performance (precision, recall, F1-score) are printed at runtime via `classification_report` and `confusion_matrix`. *(Fill in actual numbers here once you have a final run — e.g. "Best params: {'C': 10, 'gamma': 0.01}, Test accuracy: 0.97")*

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python svm.py
```

## Notes / Future Improvements

- Features are on different scales (e.g. area vs. smoothness); adding `StandardScaler` in a pipeline before fitting SVC would likely improve results, since SVM is distance-sensitive.
- The grid search only tunes `C` and `gamma` for the default RBF kernel — could extend to compare `linear` and `poly` kernels.
- Could add cross-validation score reporting and ROC-AUC as additional evaluation metrics.

## License

This project is for educational/portfolio purposes.
