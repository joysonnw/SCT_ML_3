import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog

IMG_SIZE = 128  
DATA_DIR = r"C:\Users\nwjoy\OneDrive\Documents\SkillCraft\SCT-3\PetImages"  

def load_data(data_dir):
    X, y = [], []
    for label, cls in enumerate(["Cat", "Dog"]):  # PetImages uses 'Cat' and 'Dog'
        folder = os.path.join(data_dir, cls)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Extract HOG features
            features, _ = hog(img, 
                              orientations=9, 
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=True)
            
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


print("Loading data.")
X, y = load_data(DATA_DIR)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training LinearSVC.")
base_svm = LinearSVC(max_iter=5000, verbose=1)

svm = CalibratedClassifierCV(base_svm)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
