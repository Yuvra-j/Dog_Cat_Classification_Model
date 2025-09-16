import cv2
import numpy as np
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt

def preprocess_single_image(img_path, img_size=(64, 64)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
    img = cv2.resize(img, img_size)
    
    try:
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                          channel_axis=-1, visualize=True)
    except TypeError:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, _ = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    features = features.reshape(1, -1)
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    return features_pca

def predict_image(model, img_path):
    X_real_pca = preprocess_single_image(img_path)
    prediction = model.predict(X_real_pca)
    class_name = "Dog" if prediction[0] == 1 else "Cat"
    probability = model.decision_function(X_real_pca)[0]
    print(f"Predicted class: {class_name}")
    print(f"Decision function value: {probability:.4f}")
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {class_name}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    svm_model = joblib.load('svm_model.pkl')
    real_image_path = "C:/Users/Yuvraj/OneDrive/Desktop/dataset/cat_and_dog/validation/dogs/dog.6440.jpg"
    predict_image(svm_model, real_image_path)