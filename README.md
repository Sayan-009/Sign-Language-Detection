# 🤟 Sign Language Recognition System

An end-to-end Machine Learning application that detects hand gestures via webcam and translates them into text and speech in real-time. This project is designed to bridge communication gaps for individuals using American Sign Language (ASL).

## 🚀 [Live Demo](INSERT_YOUR_STREAMLIT_LINK_HERE)

---

## 📸 Features
*   **Real-time Detection:** High-frequency hand landmark tracking using MediaPipe.
*   **Audio Feedback:** Integrated Text-to-Speech (TTS) that vocalizes recognized signs.
*   **Web Interface:** Easy-to-use browser interface built with Streamlit and WebRTC.
*   **High Accuracy:** Trained using a Random Forest classifier on normalized hand coordinates.

## 🛠️ Tech Stack
*   **Language:** Python 3.11+
*   **Computer Vision:** OpenCV, MediaPipe
*   **Machine Learning:** Scikit-learn, NumPy, Pandas
*   **Web Framework:** Streamlit, Streamlit-WebRTC


---

## 📂 Project Structure
```text
SignLanguageProject/
├── .gitignore                # Specifies files to exclude from GitHub
├── 01_capture_images.py      # Script to capture images from webcam
├── 02_mp_data_collection.py  # Script to extract MediaPipe landmarks
├── 03_train_classifier.py    # Script to train the Random Forest model
├── 04_performance.py         # Script to evaluate model accuracy/metrics
├── 05_inference_classifier.py# Local real-time detection with audio
├── app.py                    # Streamlit web application for deployment
├── features.npy              # Processed landmark data
├── labels.npy                # Corresponding sign labels for data
├── sign_language_model.p     # The final trained pickle model
├── y_predict.npy             # Stored prediction results (eval)
├── y_test.npy                # Stored test labels (eval)
├── README.md                 # Project documentation
└── requirements.txt          # List of project dependencies