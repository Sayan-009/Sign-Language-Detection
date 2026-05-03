import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# Load the model
model_dict = pickle.load(open('./sign_language_model.p', 'rb'))
model = model_dict['model']


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Extraction logic matching your training
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

            cv2.putText(img, f"Prediction: {predicted_character}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return img


st.set_page_config(page_title="Sign Language AI", layout="centered")
st.title("🤟 Real-time Sign Language Detector")
st.write("This AI recognizes hand signs and translates them to text.")

webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)