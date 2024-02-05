from ultralytics import YOLO
import cv2
import os

# Load yolov8 model
model = YOLO('runs/detect/train6/weights/last.pt')

# Load video
video_path = 'dr.mp4'
cap = cv2.VideoCapture(video_path)

# Get video details
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_duration = total_frames / fps
print(f"Original Video Duration: {original_duration} seconds")

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
output_video_path = 'output_video.avi'  # Change the file extension to '.avi'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

ret = True
frame_count = 0
# Read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # Detect objects and track
        results = model.track(frame, persist=True)

        # Plot results
        frame_ = results[0].plot()

        # Visualize
        cv2.imshow('frame', frame_)
        out.write(frame_)  # Write frame to the output video

        frame_count += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release video capture and writer objects
cap.release()
out.release()

# Check the total number of frames written to the output video
total_written_frames = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
output_duration = total_written_frames / fps
print(f"Output Video Duration: {output_duration} seconds")

# Close all windows
cv2.destroyAllWindows()
