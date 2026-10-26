import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def extract_features(img_path):
    depth_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None

    rgb_img = None
    try:
        parts = img_path.name.split('.')
        if len(parts) >= 2:
            frame_num = parts[-2] 
            potential_names = [f"manual_rgb_{frame_num}.png", f"manual_rgb_{frame_num}.jpg"]
            for name in potential_names:
                p = img_path.parent / name
                if p.exists():
                    rgb_img = cv2.imread(str(p))
                    break
        
        if rgb_img is None:
            old_style = Path(str(img_path).replace("raw_", "rgb_"))
            if old_style.exists():
                rgb_img = cv2.imread(str(old_style))
    except Exception:
        pass

   
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(depth_img.astype(np.uint8), None)
    
    
    if descriptors is not None and len(descriptors) > 0:
        depth_feat = np.mean(descriptors, axis=0)
       
        if depth_feat.shape[0] > 500:
            depth_feat = depth_feat[:500]
        elif depth_feat.shape[0] < 500:
            depth_feat = np.pad(depth_feat, (0, 500 - depth_feat.shape[0]))
    else:
        depth_feat = np.zeros(500)

    
    if rgb_img is not None:
        color_feat = np.array(cv2.mean(rgb_img)[:3])
    else:
        color_feat = np.array([0, 0, 0])

    return np.hstack([depth_feat, color_feat]).astype(float)

def main():
    train_path = Path("photos_train")
    X = []
    y = []
    
    if not train_path.exists():
        print("Error: photos_train folder not found!")
        return

    classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes.")

    for label in classes:
        print(f"Processing: {label}...", end=" ", flush=True)
        class_path = train_path / label
        img_files = list(class_path.glob("raw_*.png"))
        
        count = 0
        for img_path in img_files:
            features = extract_features(img_path)
            if features is not None:
                
                if features.shape == (503,):
                    X.append(features)
                    y.append(label)
                    count += 1
        print(f"Done ({count} images)")

    if len(X) == 0:
        print("No features extracted!")
        return

    
    X = np.vstack(X) 
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n--- Training on {X.shape[0]} samples ---")
    
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y_encoded)
    print("SVM Trained.")

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y_encoded)
    print("Random Forest Trained.")

    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("\nSuccess! Models saved. Run evaluate_model.py next.")

if __name__ == "__main__":
    main()