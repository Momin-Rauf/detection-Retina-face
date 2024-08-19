import cv2
from retinaface import RetinaFace
import time

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Set resolution to 640x480 for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables for FPS calculation
fps = 0
start_time = time.time()
frame_count = 0
process_every_nth_frame = 2  # Process every 2nd frame

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process only every nth frame
    if frame_count % process_every_nth_frame == 0:
        # Convert the frame to RGB for RetinaFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        faces = RetinaFace.detect_faces(rgb_frame)

        # Draw the detected faces and landmarks on the frame
        if faces:
            for face_id, face in faces.items():
                facial_area = face['facial_area']
                landmarks = face['landmarks']
                score = face['score']

                # Draw rectangle around the face
                cv2.rectangle(frame, 
                              (facial_area[0], facial_area[1]), 
                              (facial_area[2], facial_area[3]), 
                              (0, 255, 0), 2)

                # Draw landmarks
                for point in landmarks.values():
                    cv2.circle(frame, 
                               (int(point[0]), int(point[1])), 
                               3, 
                               (0, 0, 255), 
                               -1)

                # Display accuracy score
                cv2.putText(frame, 
                            f"Score: {score:.2f}", 
                            (facial_area[0], facial_area[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, 
                            (0, 255, 0), 
                            2)

    # Calculate FPS
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time

    # Display FPS on the frame
    cv2.putText(frame, 
                f"FPS: {fps:.2f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2)

    # Display the frame with detections
    cv2.imshow('Face and Landmark Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
