import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

def train_svm(X_train, y_train, X_val, y_val, n_components=300, kernel='rbf', C=1.0):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    svm_model = SVC(kernel=kernel, C=C, probability=True)
    svm_model.fit(X_train_pca, y_train)

    # Save reuseable files 
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')

    # Evaluate the trained model
    y_val_pred = svm_model.predict(X_val_pca)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    # Confusion Matrix and analysis
    cm = confusion_matrix(y_val, y_val_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks([0, 1], ['Cat', 'Dog'])
    plt.yticks([0, 1], ['Cat', 'Dog'])
    plt.show()

    return svm_model, scaler, pca

if __name__ == "__main__":
    print("Model trained and saved")