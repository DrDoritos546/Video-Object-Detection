import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import typer
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

app = typer.Typer()


class DetectorType(str, Enum):
    ORB = "orb"
    SIFT = "sift"
    HARRIS = "harris"
    HOG = "hog"


def flatten_descriptors(des, max_keypoints, descriptor_size):
    """Converts variable descriptors into a fixed-length array for SVM."""
    if des is None:
        return np.zeros(max_keypoints * descriptor_size, dtype=np.float32)

    # Flatten the descriptors found
    flat = des.flatten().astype(np.float32)

    # Pad with zeros if we found fewer than max_keypoints, or trim if more
    target_size = max_keypoints * descriptor_size
    if len(flat) < target_size:
        flat = np.concatenate([flat, np.zeros(target_size - len(flat))])
    else:
        flat = flat[:target_size]

    return flat


def extract_hog_features(img, fixed_size=(64, 64)):
    """Standard HOG process."""
    resized = cv2.resize(img, fixed_size)
    hog = cv2.HOGDescriptor(
        _winSize=fixed_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog.compute(resized).flatten().astype(np.float32)


def extract_features(img, detector_type: DetectorType):
    """Router function to handle different feature types."""
    # Convert 16-bit depth to 8-bit for ORB/SIFT compatibility
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if detector_type == DetectorType.HOG:
        return extract_hog_features(img)

    elif detector_type == DetectorType.ORB:
        # Limit to 100 features to keep the SVM vector size manageable
        detector = cv2.ORB_create(nfeatures=100)
        kp, des = detector.detectAndCompute(img, None)
        return flatten_descriptors(des, 100, 32)

    elif detector_type == DetectorType.SIFT:
        detector = cv2.SIFT_create(nfeatures=100)
        kp, des = detector.detectAndCompute(img, None)
        return flatten_descriptors(des, 100, 128)

    return None


@app.command()
def main(
    data_dir: Path = typer.Argument(..., help="Path to the training data folder"),
    detector_type: DetectorType = typer.Option(DetectorType.HOG, help="Feature detector to use"),
    model_output: Path = typer.Option("classifier_v3.pkl", help="Path to save the trained model")
):
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        return 1

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print("No class folders found.")
        return 1

    print(f"Using Detector: {detector_type.value}")
    all_features = []
    all_labels = []

    for class_idx, d in enumerate(class_dirs):
        images = [img for img in d.glob("*.png") if "depth" in img.name.lower()]
        print(f"  {d.name}: {len(images)} images")

        for img_path in images:
            # Load as IMREAD_UNCHANGED to handle 16-bit depth correctly
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None: continue

            feat = extract_features(img, detector_type)
            if feat is not None:
                all_features.append(feat)
                all_labels.append(class_idx)

    features = np.array(all_features)
    labels = np.array(all_labels)

    if len(features) == 0:
        print("No features extracted.")
        return 1

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTraining SVM (Linear Kernel)...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    # Evaluation
    predictions = svm.predict(X_val)
    y_score = svm.predict_proba(X_val)
    accuracy = np.sum(predictions == y_val) / len(y_val)
    print(f"Validation Accuracy: {accuracy:.1%}")

    # Generate ROC Curve for this specific detector
    class_names = [d.name for d in class_dirs]
    y_val_bin = label_binarize(y_val, classes=range(len(class_names)))

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {detector_type.value.upper()}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_{detector_type.value}.png', dpi=150)

    joblib.dump(svm, str(model_output))
    print(f"Saved: {model_output}")


if __name__ == "__main__":
    sys.exit(app())

"""
# Train HOG (Baseline)
pdm run train-classifier training_data --detector-type hog --model-output model_hog.pkl

# Train ORB
pdm run train-classifier training_data --detector-type orb --model-output model_orb.pkl

# Train SIFT
pdm run train-classifier training_data --detector-type sift --model-output model_sift.pkl
"""

