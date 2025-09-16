from skimage.feature import hog
import cv2
def extract_hog(img, img_size=(64, 64)):
    img = cv2.resize(img, img_size)
    try:
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                          channel_axis=-1, visualize=True)
    except TypeError:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, _ = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        
    return features