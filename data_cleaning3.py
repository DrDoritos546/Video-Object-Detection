import sys
import time
from pathlib import Path
import cv2
import numpy as np
import typer

from azure_kinect_video_player.playback_wrapper import AzureKinectPlaybackWrapper
from azure_kinect_video_player.image_scaler import map_uint16_to_uint8

app = typer.Typer()

@app.command()
def app_main(video_filename: Path = typer.Argument(..., help="The video filename"),
             realtime_wait: bool = typer.Option(True, help="Keep True to see what is happening"),
             init_threshold: int = typer.Option(600, help="Distance threshold in mm")):

    print(f"--- Launching Auto-Capture on: {video_filename} ---")
    
    output_dir = Path("photos_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Requirement: Using RGB data in addition to Depth
    playback_wrapper = AzureKinectPlaybackWrapper(
        video_filename, realtime_wait=realtime_wait, 
        auto_start=False, rgb=True, depth=True, ir=False
    )

    threshold = init_threshold
    def on_chg(val):
        nonlocal threshold
        threshold = val

    cv2.namedWindow("Auto-Capture Mode", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Depth Limit", "Auto-Capture Mode", threshold, 1500, on_chg)

    object_id = 0
    burst_count = 0
    is_detecting = False
    burst_limit = 15 
    cooldown = 0

    playback_wrapper.start()

    try:
        
        for current_rgb, current_depth, _ in playback_wrapper.grab_frame():
            if current_depth is None: break
            
            
            mask = (current_depth < threshold) & (current_depth > 0)
            near_points = np.argwhere(mask)
            
            display_img = map_uint16_to_uint8(current_depth, 0, 3000)
            display_colour = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            if len(near_points) > 3000:
                
                x, y, w, h = cv2.boundingRect(near_points[:, [1, 0]])
                
                if w > 50 and h > 50:
                    cv2.rectangle(display_colour, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if not is_detecting:
                        object_id += 1
                        burst_count = 0
                        is_detecting = True
                        print(f"\n[NEW OBJECT] ID: {object_id}")
                    
                    if burst_count < burst_limit and cooldown <= 0:
                        burst_count += 1
                        
                        depth_roi = current_depth[y:y+h, x:x+w]
                        rgb_roi = current_rgb[y:y+h, x:x+w]
                        
                        fname = f"{object_id}.{burst_count}.png"
                        
                        cv2.imwrite(str(output_dir / f"raw_{fname}"), depth_roi)
                        cv2.imwrite(str(output_dir / f"rgb_{fname}"), rgb_roi)
                        cv2.imwrite(str(output_dir / f"vis_{fname}"), map_uint16_to_uint8(depth_roi, 0, 3000))
                        
                        print(f"  Captured {fname}", end="\r")
                        cooldown = 8 
                else:
                    is_detecting = False
            else:
                if is_detecting:
                    print(f"\nObject {object_id} cleared.")
                is_detecting = False

            if cooldown > 0: cooldown -= 1
            cv2.imshow("Auto-Capture Mode", display_colour)
            if cv2.waitKey(1) in [ord("q"), 27]: break

    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        playback_wrapper.stop()

if __name__ == "__main__":
    app()