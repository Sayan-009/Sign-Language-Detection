import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import cv2
import pickle
import numpy as np
import mediapipe as mp
import os


# --- INITIALIZATION ---
# Using st.cache_resource so the model only loads ONCE
@st.cache_resource
def load_model():
    model_path = './sign_language_model.p'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)['model']


@st.cache_resource
def get_mp_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,  # Slightly higher for stability
        min_tracking_confidence=0.5
    )


model = load_model()
hands_tool = get_mp_hands()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class SignLanguageProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Check if tools loaded correctly
        if model is None or hands_tool is None:
            return frame.from_ndarray(img, format="bgr24")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_tool.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Normalization logic
            wrist_x = hand.landmark[0].x
            wrist_y = hand.landmark[0].y
            wrist_z = hand.landmark[0].z

            temp_data = []
            for lm in hand.landmark:
                temp_data.append(lm.x - wrist_x)
                temp_data.append(lm.y - wrist_y)
                temp_data.append(lm.z - wrist_z)

            try:
                prediction = model.predict(np.asarray(temp_data).reshape(1, -1))
                predicted_character = chr(65 + int(prediction[0]))

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img, hand, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                cv2.putText(img, f"ASL: {predicted_character}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            except Exception:
                pass

        return frame.from_ndarray(img, format="bgr24")


# --- UI ---
st.set_page_config(page_title="Sign Language AI", layout="wide")
st.title("🤟 ASL Real-time Detector")

if model is None:
    st.error("⚠️ Model file (sign_language_model.p) not found! Please check your GitHub repo.")
else:
    st.success("✅ Model loaded successfully.")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="asl-main",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Instructions")
    st.write("1. Allow camera access.")
    st.write("2. Ensure your hand is fully visible.")
    st.write("3. Use clear, steady gestures.")