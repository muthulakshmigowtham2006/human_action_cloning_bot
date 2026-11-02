import cv2

# Open the integrated webcam
cap = cv2.VideoCapture(0)

# Check whether the camera opened successfully
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Camera opened successfully!")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot receive frame")
        break

    # Display live camera feed
    cv2.imshow("Human Action Cloning - Camera Test", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows
