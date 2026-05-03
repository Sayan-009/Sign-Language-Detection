import os
import cv2
import time

_DIR = "RAW_IMAGES"

# Create main directory
if not os.path.exists(_DIR):
    os.makedirs(_DIR)


for i in range(26):
    folder_path = os.path.join(_DIR, str(i))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- PHASE 1: PREPARATION ---
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Ready for Sign {i}?", (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (123, 25, 70), 2)
        cv2.putText(frame, "Press 'q' to START", (150, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (55, 215, 65), 2)

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            # Small countdown before starting
            for count in range(3, 0, -1):
                temp_frame = frame.copy()
                cv2.putText(temp_frame, f"Starting in {count}...", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("Camera Feed", temp_frame)
                cv2.waitKey(1000)
            break

    # --- PHASE 2: CAPTURING ---
    print(f"Collecting frames for folder {i}...")
    for frameCount in range(300):
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)

        image_file_path = os.path.join(folder_path, f"{frameCount}.jpg")
        cv2.imwrite(image_file_path, frame)

        # Visual feedback during capture
        cv2.putText(frame, f"Frame {frameCount + 1}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 55, 155), 2)
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

print("Data Collection Complete!")