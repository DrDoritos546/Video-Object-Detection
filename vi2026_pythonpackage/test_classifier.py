import sys
from pathlib import Path
from enum import Enum
import cv2
import numpy as np
import typer
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

app = typer.Typer()


# Match the Enum from the training script
class DetectorType(str, Enum):
    ORB = "orb"
    SIFT = "sift"
    HARRIS = "harris"
    HOG = "hog"


def flatten_descriptors(des, max_keypoints, descriptor_size):
    """Converts variable descriptors into a fixed-length array for SVM."""
    if des is None:
        return np.zeros(max_keypoints * descriptor_size, dtype=np.float32)
    flat = des.flatten().astype(np.float32)
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
    """Router function to handle different feature types (Must match training logic)."""
    # Convert 16-bit depth to 8-bit for ORB/SIFT compatibility
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if detector_type == DetectorType.HOG:
        return extract_hog_features(img)

    elif detector_type == DetectorType.ORB:
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
    model_path: Path = typer.Argument(..., help="Path to the saved classifier (.pkl)"),
    test_dir: Path = typer.Argument(..., help="Path to the testing_data folder"),
    detector_type: DetectorType = typer.Option(DetectorType.HOG, help="Feature detector used for this model")
):
    """Load the saved SVM and test it with advanced metrics."""
    if not model_path.exists():
        print(f"Error: Model {model_path} not found.")
        return 1

    print(f"Loading classifier from {model_path}...")
    svm = joblib.load(str(model_path))

    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []

    print(f"\nEvaluating Unseen Test Data Using {detector_type.value.upper()}")
    for class_idx, class_dir in enumerate(class_dirs):
        images = [img for img in class_dir.glob("*.png") if "depth" in img.name.lower()]

        for image_path in images:
            img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
            if img is None: continue

            # Use the new router logic
            feat = extract_features(img, detector_type)
            if feat is None: continue

            feature = feat.reshape(1, -1)

            # Store Hard Prediction
            prediction = svm.predict(feature)[0]

            # Store Probability for ROC
            try:
                prob = svm.predict_proba(feature)[0]
                all_pred_probs.append(prob)
            except:
                pass

            all_true_labels.append(class_idx)
            all_pred_labels.append(prediction)

    # Calculations
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)

    if len(y_true) == 0:
        print("No test images found.")
        return 1

    # Metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    print("\nDetailed Performance Metrics")
    print(f"{'Class':<20} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name[:20]:<20} | {precision[i]:.2f} | {recall[i]:.2f} | {f1[i]:.2f}")

    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\nFinal Accuracy on Unseen Data: {accuracy:.1%}")

    # --- Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)

    # Print header row
    print(f"{'':>15}", end="")
    for name in class_names:
        print(f"{name[:14]:>15}", end="")
    print()

    # Print rows with data
    for i, name in enumerate(class_names):
        print(f"{name[:14]:>15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>15}", end="")
        print()

    # ROC Curve Generation
    if all_pred_probs:
        print("\n--- Generating ROC Curves ---")
        y_test_bin = label_binarize(y_true, classes=range(len(class_names)))
        y_score = np.array(all_pred_probs)

        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Test Set ROC - {detector_type.value.upper()}')
        plt.legend(loc="lower right", fontsize='small')
        plt.savefig(f'test_roc_{detector_type.value}.png', dpi=150)
        print(f"ROC curve saved to 'test_roc_{detector_type.value}.png'")
        plt.show()


if __name__ == "__main__":
    sys.exit(app())
"""
# Test HOG
pdm run test-classifier model_hog.pkl testing_data --detector-type hog

# Test ORB
pdm run test-classifier model_orb.pkl testing_data --detector-type orb

# Test SIFT
pdm run test-classifier model_sift.pkl testing_data --detector-type sift
"""
