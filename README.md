# Cat vs Dog Classifier using SVM

A simple classifier to distinguish between **cats and dogs** using **Histogram of Oriented Gradients (HOG)** features and a **Linear SVM (Support Vector Machine)**.

---

## How it works
1. Load images (grayscale, resized to 128x128).  
2. Extract HOG features.  
3. Train a Linear SVM (wrapped with `CalibratedClassifierCV`).  
4. Evaluate on test set.

---

## Results
- **Accuracy:** ~69%  
- Precision/Recall/F1: ~0.69 for both classes  
- Test size: 5000 images  

---

## Dataset
[Dog and Cat Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

---

## Dependencies
```bash
pip install numpy opencv-python scikit-learn scikit-image
```
## To Run
```bash
python svm.py
```
