import sys
import time
import threading
import queue
from pathlib import Path
from enum import Enum
from collections import deque, Counter

import cv2
import numpy as np
import typer
import joblib

import win32com.client
import pythoncom

from azure_kinect_video_player.playback_wrapper import AzureKinectPlaybackWrapper

app = typer.Typer()

# TTS
speech_queue = queue.Queue()


def tts_worker():
    """Background thread to handle speech using native Windows SAPI."""
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        text = speech_queue.get()
        if text is None:
            break
        speaker.Speak(text)
        speech_queue.task_done()
    pythoncom.CoUninitialize()


# Detector
class DetectorType(str, Enum):
    ORB = "orb"
    SIFT = "sift"
    HARRIS = "harris"
    HOG = "hog"


def flatten_descriptors(des, max_keypoints, descriptor_size):
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
    resized = cv2.resize(img, fixed_size)
    hog = cv2.HOGDescriptor(
        _winSize=fixed_size, _blockSize=(16, 16), _blockStride=(8, 8),
        _cellSize=(8, 8), _nbins=9
    )
    return hog.compute(resized).flatten().astype(np.float32)


def extract_features(img, detector_type: DetectorType):
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
    data_dir: Path = typer.Argument(..., help="Path to training_data"),
    video_path: Path = typer.Argument(..., help="Path to your test video file"),
    detector_type: DetectorType = typer.Option(DetectorType.HOG, help="Feature detector used for this model")
):
    if not model_path.exists() or not video_path.exists():
        print("Error: Model or Video file not found.")
        return 1

    print(f"Loading {detector_type.value.upper()} classifier from {model_path}...")
    svm = joblib.load(str(model_path))

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    playback_wrapper = AzureKinectPlaybackWrapper(
        video_path, realtime_wait=True, auto_start=False, rgb=True, depth=True, ir=False
    )
    cv2.namedWindow("Video Classification", cv2.WINDOW_NORMAL)

    threshold = 500
    near_threshold = 350
    first_frame = True
    out_of_sensor_mask = None
    last_spoken_name = None

    # Stabilisation Settings (Dynamic)
    prediction_history = deque(maxlen=90)  # Wait 3 seconds

    if detector_type == DetectorType.HOG:
        CONFIDENCE_THRESHOLD = 0.40  # Very lenient for HOG jitter
        STABILITY_THRESHOLD = 15  # Only needs 15 valid frames out of 90
    elif detector_type == DetectorType.SIFT:
        CONFIDENCE_THRESHOLD = 0.40  # SIFT struggles slightly with smooth depth data
        STABILITY_THRESHOLD = 20  # Only needs 20 valid frames out of 90
    else:  # ORB
        CONFIDENCE_THRESHOLD = 0.50  # ORB is highly confident
        STABILITY_THRESHOLD = 25  # Needs 25 valid frames out of 90

    playback_wrapper.start()
    print(f"Starting playback. Press 'q' to quit.")

    try:
        for colour_image, depth_image, ir_image in playback_wrapper.grab_frame():
            if colour_image is None and depth_image is None:
                break

            display_frame = colour_image.copy() if colour_image is not None else None
            smoothed_prediction = "Detecting..."
            confidence_str = ""

            if depth_image is not None:
                _, thresholded_depth_image = cv2.threshold(depth_image, threshold, 65535, cv2.THRESH_BINARY)
                thresholded_depth_image[-80:, :] = 65535
                thresholded_depth_image[:, :150] = 65535

                if first_frame:
                    out_of_sensor_mask = np.ma.masked_where(thresholded_depth_image == 0, thresholded_depth_image)
                    first_frame = False

                too_close_mask = (depth_image > 0) & (depth_image < near_threshold)
                thresholded_depth_image[too_close_mask] = 65535

                if out_of_sensor_mask is not None:
                    points = np.argwhere(thresholded_depth_image == 0)
                    if len(points) > 0:
                        bb_points = points[~out_of_sensor_mask.mask[points[:, 0], points[:, 1]]]

                        if len(bb_points) > 0:
                            y, x, h, w = cv2.boundingRect(bb_points)

                            if 0 < w < depth_image.shape[1] and 0 < h < depth_image.shape[0]:
                                cropped_depth = depth_image[y:y + h, x:x + w]

                                feat = extract_features(cropped_depth, detector_type)
                                if feat is not None:
                                    feature = feat.reshape(1, -1)

                                    # Get Probability
                                    raw_pred_idx = svm.predict(feature)[0]
                                    raw_pred_name = class_names[raw_pred_idx]

                                    if hasattr(svm, "predict_proba"):
                                        probs = svm.predict_proba(feature)[0]
                                        max_prob = probs[raw_pred_idx]

                                        if max_prob >= CONFIDENCE_THRESHOLD:
                                            prediction_history.append(raw_pred_name)
                                        else:
                                            prediction_history.append("Uncertain")
                                    else:
                                        prediction_history.append(raw_pred_name)

                                    # 2. Majority Vote
                                    if len(prediction_history) == prediction_history.maxlen:
                                        counts = Counter(prediction_history)

                                        # Stop uncertain
                                        if "Uncertain" in counts:
                                            del counts["Uncertain"]

                                        # Check Stability
                                        if counts:
                                            most_common_pred, count = counts.most_common(1)[0]

                                            if count >= STABILITY_THRESHOLD:
                                                smoothed_prediction = most_common_pred
                                                confidence_str = f" (Stable: {count}/{prediction_history.maxlen})"

                                                # TTS
                                                if smoothed_prediction != last_spoken_name:
                                                    clean_name = smoothed_prediction.split("_")[
                                                        -1] if "_" in smoothed_prediction else smoothed_prediction
                                                    speech_queue.put(f"I am certain I see a {clean_name}")
                                                    last_spoken_name = smoothed_prediction
                                            else:
                                                smoothed_prediction = "Stabilising..."
                                        else:
                                            smoothed_prediction = "Stabilising..."

                                # Visuals
                                if display_frame is not None:
                                    sx, sy = 3.0, 1.875
                                    x_rgb, y_rgb = int(x * sx), int(y * sy)
                                    w_rgb, h_rgb = int(w * sx), int(h * sy)

                                    box_colour = (0, 255, 0) if "Stable" in confidence_str else (0, 165, 255)
                                    cv2.rectangle(display_frame, (x_rgb, y_rgb), (x_rgb + w_rgb, y_rgb + h_rgb),
                                                  box_colour, 2)

                                    label = f"{smoothed_prediction}{confidence_str}"
                                    cv2.rectangle(display_frame, (x_rgb, y_rgb - 30), (x_rgb + len(label) * 11, y_rgb),
                                                  box_colour, -1)
                                    cv2.putText(display_frame, label, (x_rgb + 5, y_rgb - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, (0, 0, 0), 2)

            if display_frame is not None:
                cv2.imshow("Video Classification", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    speech_queue.put(None)
    cv2.destroyAllWindows()
    playback_wrapper.stop()


if __name__ == "__main__":
    sys.exit(app())


"""
COMMANDS TO RUN TTS:
# pdm run run-classifier model_orb.pkl training_data kinect-testing-set.mkv --detector-type orb

CHANGE model_orb.pkl and --detector-type orb

"""
