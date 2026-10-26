import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


from extract_and_train import extract_features

def main():
    test_path = Path("photos_test")
    
    try:
        with open('svm_model.pkl', 'rb') as f:
            svm = pickle.load(f)
        with open('rf_model.pkl', 'rb') as f:
            rf = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model files not found. Run extract_and_train.py first!")
        return

    X_test = []
    y_test = []
    
    classes = le.classes_
    print(f"Evaluating models against {len(classes)} classes...")

    for label in classes:
        class_path = test_path / label
        if not class_path.exists():
            print(f"Warning: Folder for {label} not found in photos_test. Skipping.")
            continue
            
        img_files = list(class_path.glob("raw_*.png"))
        print(f"Processing {label}: {len(img_files)} images found.")
        
        for img_path in img_files:
            features = extract_features(img_path)
            if features is not None and features.shape == (503,):
                X_test.append(features)
                y_test.append(label)

    if not X_test:
        print("No test features extracted. Check your photos_test folder!")
        return

    X_test = np.vstack(X_test)
    y_test_encoded = le.transform(y_test)

    
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test_encoded, y_pred_svm)
    
    
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test_encoded, y_pred_rf)

    print("\n" + "="*30)
    print(f"SVM Accuracy: {acc_svm:.2%}")
    print(f"RF Accuracy:  {acc_rf:.2%}")
    print("="*30 + "\n")

    print("Detailed Report (Random Forest):")
    print(classification_report(y_test_encoded, y_pred_rf, target_names=classes))

    
    cm = confusion_matrix(y_test_encoded, y_pred_rf)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

if __name__ == "__main__":
    main()