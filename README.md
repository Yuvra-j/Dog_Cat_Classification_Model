"Cat-Dog Classification Model'

Binary classifier using SVM with HOG features and PCA to distinguish cats and dogs from images. Trained on Kaggle's Dogs vs. Cats dataset.
- HOG feature extraction.
- PCA for dimensionality reduction.
- SVM classifier.
- Single image prediction.

Installation:
1. Clone the repository: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`

Usage:
- Load data: `python data/load_data.py`
- Train model: `python models/model.py`
- Evaluate: `python evaluation_of_model/evaluate.py`
- Predict single image: `python prediction/test.py`

Results:
- Accuracy:79.89
- Classification report included in notebook.

DataSet:
https://www.kaggle.com/c/dogs-vs-cats/data
