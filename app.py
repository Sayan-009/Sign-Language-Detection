import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# 1. Load the model
model_dict = pickle.load(open('./sign_language_model.p', 'rb'))
model = model_dict['model']


# 2. Updated Processor (Modern WebRTC approach)
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Wrist-relative normalization (Matches your training)
            wrist_x = hand.landmark[0].x
            wrist_y = hand.landmark[0].y
            wrist_z = hand.landmark[0].z

            temp_data = []
            for lm in hand.landmark:
                temp_data.append(lm.x - wrist_x)
                temp_data.append(lm.y - wrist_y)
                temp_data.append(lm.z - wrist_z)

            prediction = model.predict(np.asarray(temp_data).reshape(1, -1))
            predicted_character = chr(65 + int(prediction[0]))

            # Draw landmarks on screen
            mp_drawing.draw_landmarks(
                img, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            cv2.putText(img, f"Prediction: {predicted_character}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")


# --- UI LAYOUT ---
st.set_page_config(page_title="Sign Language AI", layout="centered")
st.title("🤟 Real-time Sign Language Detector")
st.write("Hold your hand in front of the camera to translate signs.")

# Centering the camera window
col1, col2, col3 = st.columns([0.5, 3, 0.5])

with col2:
    webrtc_streamer(
        key="sign-lang-main",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,  # Using Processor instead of Transformer
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

st.info("Ensure you have allowed camera access in your browser settings.")