import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

gt = []
with open("rubik_dataset/gt.txt", 'r') as f:
  for line in f:
    values = line.strip().split(',')
    values = [int(float(v)) for v in values]
    gt.append(values)

video_path = 'rubik_dataset/rubik.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  raise RuntimeError("Failed to open video file")

ret, prev_frame = cap.read()
if not ret:
  raise RuntimeError("Failed to read first frame")

ST_PARAMS = dict(
  maxCorners=10,
  qualityLevel=0.3,
  minDistance=7,
  blockSize=7
)

LK_PARAMS = dict(
  winSize=(20, 20),
  maxLevel=2,
  criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03)
)

x, y, w, h = map(int, gt[0][:4])

frame_with_bbox = prev_frame.copy()
cv2.rectangle(frame_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Display the annotated frame
plt.imshow(cv2.cvtColor(frame_with_bbox, cv2.COLOR_BGR2RGB))
plt.title("Initial Frame with Bounding Box")
plt.show()

# Crop the ROI and convert to grayscale
roi = prev_frame[y:y+h, x:x+w]
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# Detect features in the entire frame
all_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **ST_PARAMS)

# Filter features inside the initial bounding box
selected_points = []
for point in all_points:
  px, py = point.ravel()
  if x <= px <= x+w and y <= py <= y+h:
    selected_points.append(point)
if not selected_points:
  raise RuntimeError("No features found in the initial bounding box!")
prevPts = np.array(selected_points, dtype=np.float32) # Format for tracking

# Get video properties from the input
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output file and codec (MP4 format)
output_path = 'rubik_coba.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4 format

# Initialize VideoWriter
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create a black mask (same size as input frames)
mask = np.zeros_like(prev_frame)

# Track frame count for debugging/analysis
frame_count = 0

import time
start_time = time.time() # Track processing start time

while True:
  # Read next frame
  ret, frame = cap.read()
  if not ret: # Exit loop if video ends
    break
  # Convert frame to grayscale (required for optical flow)
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  if prevPts is not None:
    # Calculate optical flow between previous and current frame
    nextPts, status, _ = cv2.calcOpticalFlowPyrLK(
      prev_gray, frame_gray, prevPts, None, **LK_PARAMS)
    if nextPts is not None:
      # Filter only successfully tracked points
      good_new = nextPts[status == 1]
      good_prev = prevPts[status == 1]
      # Draw tracking markers
      for (new, prev) in zip(good_new, good_prev):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()

        # Convert to integers for drawing functions
        x_new, y_new = map(int, (x_new, y_new))
        x_prev, y_prev = map(int, (x_prev, y_prev))

        # Draw current position (green dot)
        frame = cv2.circle(frame, (x_new, y_new), 5, (0, 255, 0), -1)

      # Combine frame with motion trail mask
      visualized_frame = cv2.add(frame, mask)

      # Calculate and display real-time FPS
      elapsed_time = time.time() - start_time
      fps = frame_count / elapsed_time if elapsed_time > 0 else 0
      cv2.putText(
      visualized_frame,
      f"FPS: {fps:.2f}",
      (width - 200, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      1, (255, 0, 0), 2, cv2.LINE_AA
      )

      # Write frame to output video
      out.write(visualized_frame)

      # Update points for next iteration
      prevPts = good_new.reshape(-1, 1, 2)

    else:
      # If tracking fails, write original frame without annotations
      out.write(frame)
      prevPts = None
  else:
    # No points to track (initial failure case)
    out.write(frame)

  # Update previous frame reference
  prev_gray = frame_gray.copy()
  frame_count += 1 # Increment frame counter

# Release video capture and writer objects
cap.release() # Close input video file
out.release() # Finalize output video writing
# Close all OpenCV windows
cv2.destroyAllWindows()
# Print completion message with output path
print(f"Processing complete. Output saved to {output_path}")