import cv2
import os
import time

def collect_data(video_source, output_dir, duration, interval):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    date = time.strftime("%Y%m%d-%H%M%S")
    cap = cv2.VideoCapture(video_source, cv2.CAP_AVFOUNDATION)
    start_time = time.time()
    next_capture_time = start_time + 10

    frame_count = 0
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        
        # Check if it's time to capture the next frame
        if current_time >= next_capture_time:
            cv2.imwrite(os.path.join(output_dir, f"frame_{date}_{frame_count}.jpg"), frame)
            print(f"Saved frame #{frame_count} at {current_time - start_time:.4f}s")
            frame_count += 1
            next_capture_time = start_time + (frame_count + 1) * interval

        cv2.putText(frame, f'Time: {(current_time - start_time):.4f}s Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    

if __name__ == "__main__":
    video_source = 1  # Change to video file path for file input
    output_dir = "/Users/benjamin/Documents/plushie dataset/opencv"
    duration = 100  # Collect data for 100 seconds
    interval = 3  # Capture frame every 3 seconds

    collect_data(video_source, output_dir, duration, interval)