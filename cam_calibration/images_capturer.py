import cv2
import time
import os

# Create a folder to save images
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Starting live capture... Press 'q' to quit early.")

image_count = 0
start_time = time.time()

while image_count < 20:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Show the frame
    cv2.imshow("Webcam Feed", frame)

    # Check if 5 seconds have passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= 2:
        image_count += 1
        filename = os.path.join(save_dir, f"image_{image_count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        start_time = time.time()  # Reset the timer

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture interrupted by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Image capture complete.")
