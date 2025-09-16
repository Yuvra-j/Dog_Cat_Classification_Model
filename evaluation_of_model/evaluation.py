from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, X_test_pca, y_test):
    y_pred = model.predict(X_test_pca)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks([0, 1], ['Cat', 'Dog'])
    plt.yticks([0, 1], ['Cat', 'Dog'])
    plt.show()

if __name__ == "__main__":
    print("Evaluation complete")