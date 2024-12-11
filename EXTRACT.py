import cv2  # OpenCV for video and image processing
import numpy as np  # NumPy for numerical operations

# Load a video file (ensure the video file is in the same directory or provide a full path)
cap = cv2.VideoCapture("video2.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Video file not found or unable to open.")

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame of the video.")

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using the Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of the optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image to represent motion
    hsv = np.zeros_like(frame)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents speed

    # Convert HSV image to BGR for display
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display the optical flow
    cv2.imshow('Optical Flow', rgb_flow)

    # Update the previous frame
    prev_gray = gray

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()