import os
import cv2
import pickle
import mediapipe as mp
import numpy as np

# --- INITIALIZATION ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 1. Load Model ONCE
model_dict = pickle.load(open('./sign_language_model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success: break


        # Prepare image for MediaPipe
        image = cv2.flip(image, 1)  # Flip early for natural movement
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        temp_data = []

        if results.multi_hand_landmarks:
            # Use the same extraction logic as training
            hand = results.multi_hand_landmarks[0]
            wrist_x, wrist_y, wrist_z = hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z

            # Draw Landmarks
            mp_drawing.draw_landmarks(
                image, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


            for lm in hand.landmark:
                temp_data.append(lm.x - wrist_x)
                temp_data.append(lm.y - wrist_y)
                temp_data.append(lm.z - wrist_z)

            # --- PREDICTION ---
            prediction = model.predict(np.asarray(temp_data).reshape(1, -1))


            predicted_character = chr(65 + int(prediction[0]))

            cv2.putText(image, f"Sign: {predicted_character}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Sign Language Detector', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()