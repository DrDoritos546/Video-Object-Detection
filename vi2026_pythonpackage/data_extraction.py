import sys
import time
from pathlib import Path
import cv2
import numpy as np
import typer
from enum import Enum

from azure_kinect_video_player.playback_wrapper import AzureKinectPlaybackWrapper
from azure_kinect_video_player.image_scaler import map_uint16_to_uint8

app = typer.Typer()

output_dir = Path("testing_data")
output_dir.mkdir(parents=True, exist_ok=True)


# Define the Enum for the command line
class DetectorType(str, Enum):
    NONE = "none"
    ORB = "orb"
    HARRIS = "harris"
    SIFT = "sift"
    HOG = "hog"


@app.command()
def app_main(video_filename: Path = typer.Argument(..., help="The video filename"),
             detector_type: DetectorType = typer.Option(DetectorType.NONE, help="Feature detector to use"),
             realtime_wait: bool = typer.Option(True, help="Wait for the next frame to be displayed"),
             rgb: bool = typer.Option(True, help="Display RGB image"),
             depth: bool = typer.Option(True, help="Display depth image"),
             ir: bool = typer.Option(False, help="Display IR image"),
             init_threshold: int = typer.Option(500, help="Initial threshold value")):
    playback_wrapper = AzureKinectPlaybackWrapper(video_filename, realtime_wait=realtime_wait, auto_start=False,
                                                  rgb=rgb, depth=depth, ir=ir)

    if rgb: cv2.namedWindow("Colour", cv2.WINDOW_NORMAL)
    if depth: cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    threshold = init_threshold

    def on_threshold_change(value):
        nonlocal threshold
        threshold = value

    threshold_win_name = "Thresholded Depth"
    cv2.namedWindow(threshold_win_name)
    cv2.createTrackbar("Threshold", threshold_win_name, threshold, 1500, on_threshold_change)

    first_frame = True
    out_of_sensor_mask = None

    near_threshold = 350

    def on_near_threshold_change(value):
        nonlocal near_threshold
        near_threshold = value

    cv2.createTrackbar("Near Threshold", threshold_win_name, near_threshold, 1500, on_near_threshold_change)

    capture_end_time = 0
    current_class_dir = None

    class_keybinds = { # Keybinds for 3 second captures
        ord('1'): 'mug', ord('2'): 'illuminator', ord('3'): 'baby', ord('4'): 'cup-noodle',
        ord('5'): 'measuring-tape', ord('6'): 'dinosaur', ord('7'): 'Einstein', ord('8'): 'hour-glass',
        ord('9'): 'label-printer', ord('0'): 'spray', ord('-'): 'dog', ord('='): 'robot'
    }

    # Initialise the chosen detector before the loop
    orb = cv2.ORB_create(nfeatures=100) if detector_type == DetectorType.ORB else None
    sift = cv2.SIFT_create() if detector_type == DetectorType.SIFT else None

    start_time = time.time()
    playback_wrapper.start()

    try:
        frame_count = 0
        for colour_image, depth_image, ir_image in playback_wrapper.grab_frame():
            frame_count += 1
            if colour_image is None and depth_image is None:
                break

            if colour_image is not None:
                colour_image = colour_image.copy()

            cropped_rgb = None
            cropped_depth = None

            if depth_image is not None:
                _, thresholded_depth_image = cv2.threshold(depth_image, threshold, 65535, cv2.THRESH_BINARY)

                # Cutoffs to remove the background and arm
                bottom_cutoff = 80
                thresholded_depth_image[-bottom_cutoff:, :] = 65535
                left_cutoff = 150
                thresholded_depth_image[:, :left_cutoff] = 65535

                if first_frame:
                    out_of_sensor_mask = np.ma.masked_where(thresholded_depth_image == 0, thresholded_depth_image)
                    first_frame = False

                depth_image_8bit = map_uint16_to_uint8(depth_image, 0, 10000)
                depth_image_colour = cv2.cvtColor(depth_image_8bit, cv2.COLOR_GRAY2BGR)

                if out_of_sensor_mask is not None:
                    depth_image_colour[out_of_sensor_mask.mask] = (0, 0, 255)

                too_close_mask = (depth_image > 0) & (depth_image < near_threshold)
                thresholded_depth_image[too_close_mask] = 65535

                if out_of_sensor_mask is not None:
                    points = np.argwhere(thresholded_depth_image == 0)
                    if len(points) > 0:
                        bb_points = points[~out_of_sensor_mask.mask[points[:, 0], points[:, 1]]]
                        if len(bb_points) > 0:
                            y, x, h, w = cv2.boundingRect(bb_points)

                            if 0 < w < depth_image.shape[1] and 0 < h < depth_image.shape[0]:
                                cv2.rectangle(depth_image_colour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cropped_depth = depth_image[y:y + h, x:x + w]

                                sx, sy = 3.0, 1.875
                                x_rgb, y_rgb = int(x * sx), int(y * sy)
                                w_rgb, h_rgb = int(w * sx), int(h * sy)

                                if colour_image is not None:
                                    cv2.rectangle(colour_image, (x_rgb, y_rgb), (x_rgb + w_rgb, y_rgb + h_rgb),
                                                  (0, 255, 0), 2)
                                    y1, y2 = max(0, y_rgb), min(colour_image.shape[0], y_rgb + h_rgb)
                                    x1, x2 = max(0, x_rgb), min(colour_image.shape[1], x_rgb + w_rgb)
                                    cropped_rgb = colour_image[y1:y2, x1:x2]

                                    # Live Visualisation
                                    if cropped_rgb is not None and detector_type != DetectorType.NONE:
                                        grey_crop = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2GRAY)
                                        feature_vis = None

                                        # Ensure crop is large enough to prevent OpenCV crashes
                                        if grey_crop.shape[0] >= 32 and grey_crop.shape[1] >= 32:
                                            if detector_type == DetectorType.ORB:
                                                kp, _ = orb.detectAndCompute(grey_crop, None)
                                                feature_vis = cv2.drawKeypoints(cropped_rgb, kp, None,
                                                                                color=(0, 255, 0),
                                                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                                            elif detector_type == DetectorType.SIFT:
                                                kp, _ = sift.detectAndCompute(grey_crop, None)
                                                feature_vis = cv2.drawKeypoints(cropped_rgb, kp, None,
                                                                                color=(0, 255, 0),
                                                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                                            elif detector_type == DetectorType.HARRIS:
                                                grey_float = np.float32(grey_crop)
                                                corners = cv2.cornerHarris(grey_float, 2, 3, 0.04)
                                                corners_dilated = cv2.dilate(corners, None)
                                                feature_vis = cropped_rgb.copy()
                                                feature_vis[corners_dilated > 0.01 * corners_dilated.max()] = [0, 255,
                                                                                                               0]

                                        # HOG visualizer (needs slightly different size logic)
                                        if detector_type == DetectorType.HOG and grey_crop.shape[0] >= 16 and \
                                            grey_crop.shape[1] >= 16:
                                            cell_size = (8, 8)
                                            block_size = (2, 2)
                                            nbins = 9

                                            win_size = (
                                                grey_crop.shape[1] // cell_size[1] * cell_size[1],
                                                grey_crop.shape[0] // cell_size[0] * cell_size[0]
                                            )
                                            hog = cv2.HOGDescriptor(
                                                _winSize=win_size,
                                                _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
                                                _blockStride=(cell_size[1], cell_size[0]),
                                                _cellSize=(cell_size[1], cell_size[0]),
                                                _nbins=nbins
                                            )
                                            hog_features_cv = hog.compute(grey_crop)

                                            if hog_features_cv is not None:
                                                n_cells = (win_size[1] // cell_size[0], win_size[0] // cell_size[1])
                                                try:
                                                    hog_features = hog_features_cv.reshape(
                                                        n_cells[1] - block_size[1] + 1,
                                                        n_cells[0] - block_size[0] + 1,
                                                        block_size[0], block_size[1], nbins
                                                    ).transpose((1, 0, 2, 3, 4))

                                                    gradients = np.zeros((n_cells[0], n_cells[1], nbins))
                                                    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

                                                    for off_y in range(block_size[0]):
                                                        for off_x in range(block_size[1]):
                                                            gradients[
                                                                off_y:n_cells[0] - block_size[0] + off_y + 1, off_x:
                                                                                                              n_cells[
                                                                                                                  1] -
                                                                                                              block_size[
                                                                                                                  1] + off_x + 1] += \
                                                            hog_features[:, :, off_y, off_x, :]
                                                            cell_count[
                                                                off_y:n_cells[0] - block_size[0] + off_y + 1, off_x:
                                                                                                              n_cells[
                                                                                                                  1] -
                                                                                                              block_size[
                                                                                                                  1] + off_x + 1] += 1

                                                    gradients /= cell_count

                                                    # Display orientation bin 5

                                                    show_image = gradients[:, :, 5]
                                                    cv2.normalize(show_image, show_image, 0, 255, cv2.NORM_MINMAX)
                                                    show_image = show_image.astype(np.uint8)

                                                    feature_vis = cv2.resize(show_image, (show_image.shape[1] * 4,
                                                                                          show_image.shape[0] * 4),
                                                                             interpolation=cv2.INTER_NEAREST)
                                                except Exception:
                                                    pass

                                        if feature_vis is not None:
                                            cv2.imshow("Features", feature_vis)

                cv2.imshow(threshold_win_name, thresholded_depth_image)
                cv2.imshow("Depth", depth_image_colour)

            if rgb and colour_image is not None: cv2.imshow("Colour", colour_image)

            key = cv2.waitKey(1)
            if key == ord("q") or key == 27: break

            if key in class_keybinds:
                # 3 seconds of frame capture for training data
                class_name = class_keybinds[key]
                current_class_dir = output_dir / class_name
                current_class_dir.mkdir(parents=True, exist_ok=True)
                capture_end_time = time.time() + 3.0
                print(f"3-second burst capture for '{class_name}'")

            if time.time() < capture_end_time and current_class_dir is not None:
                if cropped_rgb is not None and cropped_depth is not None:
                    cv2.imwrite(str(current_class_dir / f"rgb_{frame_count:06d}.png"), cropped_rgb)
                    cv2.imwrite(str(current_class_dir / f"depth_{frame_count:06d}.png"), cropped_depth)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    playback_wrapper.stop()
    return 0


if __name__ == "__main__":
    sys.exit(app())
    # cd C:\Users\bee21\OneDrive\Desktop\VI2026-PythonPackage
    # pdm run data-extraction "C:\Users\bee21\OneDrive\Desktop\Kinect Coursework Data 2025-2026\kinect-training-set.mkv"
    # pdm run train_classifier training_data
    # pdm run run-classifier classifier_v3.pkl training_data "C:\Users\bee21\OneDrive\Desktop\Kinect Coursework Data 2025-2026\kinect-testing-set.mkv"
