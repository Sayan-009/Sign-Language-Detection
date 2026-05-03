import os
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


_DIR = "RAW_IMAGES"

data = []
labels = []

NUM_HANDS = 1
LANDMARKS_PER_HAND = 21
COORDINATES = 3

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=0.5) as hands:
    for image_folder in os.listdir(_DIR):
        for image in os.listdir(os.path.join(_DIR, image_folder)):
            temp_data = []

            bgr_image = cv2.imread(os.path.join(_DIR, image_folder, image))
            if bgr_image is None: continue

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                wrist_x = hand.landmark[0].x
                wrist_y = hand.landmark[0].y
                wrist_z = hand.landmark[0].z
                for lm in hand.landmark:
                    temp_data.append(lm.x - wrist_x)
                    temp_data.append(lm.y - wrist_y)
                    temp_data.append(lm.z - wrist_z)

            else:
                # FAILSAFE: Fill with zeros if no hand found (63 values)
                temp_data = [0] * (LANDMARKS_PER_HAND * COORDINATES)


            data.append(temp_data)
            labels.append(image_folder)


data_np = np.array(data)
labels_np = np.array(labels)

# Save the final features
np.save('features.npy', data_np)
np.save('labels.npy', labels_np)
print("Landmark extraction successful. Shape:", data_np.shape)