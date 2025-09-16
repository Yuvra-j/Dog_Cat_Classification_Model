import os
import cv2
import numpy as np
from skimage.feature import hog

def load_and_preprocess(folder, label, img_size=(64, 64)):
    data = []
    labels = []
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)  
        if img is None:
            print(f"Skipping corrupted file: {img_path}")
            continue  
        img = cv2.resize(img, img_size)

        try:
            features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                              channel_axis=-1, visualize=True)
        except TypeError:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features, _ = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        
        data.append(features)
        labels.append(label)  # 0 for cat, 1 for dog
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    train_cats_data, train_cats_labels = load_and_preprocess("C:/Users/Yuvraj/OneDrive/Desktop/dataset/cat_and_dog/train2/cats", 0)
    print(f"Loaded {len(train_cats_data)} cat images")