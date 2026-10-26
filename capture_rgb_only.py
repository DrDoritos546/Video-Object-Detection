import sys
from pathlib import Path
import cv2
import numpy as np
import typer
from azure_kinect_video_player.playback_wrapper import AzureKinectPlaybackWrapper

app = typer.Typer()

@app.command()
def main(video_filename: Path = typer.Argument(..., help="The video filename"),
         output_folder: Path = typer.Option("photos_train", help="Folder to save RGB crops"),
         init_threshold: int = typer.Option(600, help="Depth threshold in mm")):

    playback_wrapper = AzureKinectPlaybackWrapper(
        video_filename, realtime_wait=True, 
        auto_start=False, rgb=True, depth=True, ir=False
    )

    threshold = init_threshold
    offset_x = 200 
    offset_y = 200
    is_paused = False 

    def on_thresh(val): nonlocal threshold; threshold = val
    def on_offset_x(val): nonlocal offset_x; offset_x = val - 500 
    def on_offset_y(val): nonlocal offset_y; offset_y = val - 500

    cv2.namedWindow("RGB Aligner", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Depth Limit", "RGB Aligner", threshold, 1500, on_thresh)
    cv2.createTrackbar("Shift X", "RGB Aligner", 500, 1500, on_offset_x)
    cv2.createTrackbar("Shift Y", "RGB Aligner", 500, 1000, on_offset_y)

    save_count = 0
    playback_wrapper.start()
    frame_generator = playback_wrapper.grab_frame()

    print("\n--- MANUAL CAPTURE MODE ---")
    print(" 'p' : Pause/Play")
    print(" 's' : Save current RGB crop")
    print(" 'q' : Quit")

    try:
        current_rgb, current_depth = None, None
        
        while True:
            if not is_paused:
                try:
                    current_rgb, current_depth, _ = next(frame_generator)
                except StopIteration:
                    print("\nEnd of video reached.")
                    break
            
            if current_depth is None: continue
            
            mask = (current_depth < threshold) & (current_depth > 0)
            near_points = np.argwhere(mask)
            display_rgb = current_rgb.copy()


            if len(near_points) > 3000:
                x, y, w, h = cv2.boundingRect(near_points[:, [1, 0]])
                
                
                rgb_x = max(0, min(current_rgb.shape[1] - w, x + offset_x))
                rgb_y = max(0, min(current_rgb.shape[0] - h, y + offset_y))
                
                
                cv2.rectangle(display_rgb, (rgb_x, rgb_y), (rgb_x+w, rgb_y+h), (0, 255, 0), 2)
                
                rgb_roi = current_rgb[rgb_y : rgb_y+h, rgb_x : rgb_x+w]
            else:
                rgb_roi = None

            if is_paused:
                cv2.putText(display_rgb, "PAUSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow("RGB Aligner", display_rgb)
            
            key = cv2.waitKey(1) & 0xFF
            
          
            if key == ord('s'):
                if rgb_roi is not None:
                    save_count += 1
                    
                    save_path = Path(output_folder) / f"manual_rgb_{save_count}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path), rgb_roi)
                    print(f"Saved Image #{save_count}: {save_path.name}")
                else:
                    print("No object detected in depth! Adjust Depth Limit.")

            elif key == ord('p'):
                is_paused = not is_paused
            elif key == ord('q') or key == 27:
                break

    finally:
        playback_wrapper.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app()