import cv2
import numpy as np
import typer
from pathlib import Path

def main(video_path: Path = typer.Argument(..., help="Path to the .mkv video")):
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("\n--- Controls ---")
    print("Space: Pause/Play")
    print("Right Arrow: Skip +5s | Left Arrow: Back -5s")
    print("Up Arrow: Faster | Down Arrow: Slower")
    print("Q: Quit")

    delay = 30 
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
               
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

     
        depth_map = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(depth_map, 30, 150, cv2.THRESH_BINARY)
        clean_object = cv2.bitwise_and(depth_map, depth_map, mask=mask)

        cv2.imshow("Clean Object View (AI Input)", clean_object)

      
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):  
            paused = not paused
        elif key == 83:  
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 150) 
        elif key == 81:
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 150))
        elif key == 82: 
            delay = max(1, delay - 5)
        elif key == 84:  
            delay += 5

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    typer.run(main)