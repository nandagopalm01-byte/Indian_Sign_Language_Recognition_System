import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from tensorflow.keras.models import load_model

# ===========================
# Loading model
# ===========================
MODEL_PATH = r"final_exported_model04.h5"
model = load_model(MODEL_PATH, compile=False)

classes = list("abcdefghijklmnopqrstuvwxyz")
IMG_SIZE = 128

st.title("ASL Hand Gesture Recognition - IP Camera")
st.write("Show a hand sign to the camera. Press **Start** to begin and **Stop** to end.")

# ===========================
# CVZONE Hand Detector
# ===========================
detector = HandDetector(maxHands=1)

# ===========================
# UI controls
# ===========================
run = st.checkbox("Start Camera")
stop_button = st.button("Stop Camera")
frame_window = st.image([])
prediction_text = st.empty()


IP_URL = "http://192.168.1.3:8080/video"  

if "video" not in st.session_state:
    st.session_state.video = None

# ===========================
# STOP BUTTON HANDLING - EXIT CLEANLY
# ===========================
if stop_button:
    if st.session_state.video is not None:
        st.session_state.video.release()
        st.session_state.video = None

    st.write("### **Camera stopped. Refresh page to restart.**")
    st.stop()

# ===========================
# CAMERA LOOP
# ===========================
if run:
    if st.session_state.video is None:
        st.session_state.video = cv2.VideoCapture(0) # if IP CAMERA use 'URL'

    while True:
        success, img = st.session_state.video.read()
        if not success:
            st.warning("Failed to read from phone camera.")
            break

        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img)  # detect hands

        prediction_label = ""

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']  # bounding box

            crop = img[y:y+h, x:x+w]

            if crop.size != 0:
                hand_img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) # reshapes to 128,128
                hand_img = hand_img.astype("float32") / 255.0 # normalize the pixel values of the image to a standard range of \(0.0,1.0\)
                hand_img = np.expand_dims(hand_img, axis=0) # hanges the shape to (1, height, width, channels)

                preds = model.predict(hand_img, verbose=0)
                prediction_label = classes[np.argmax(preds)]

        cv2.putText(img, f"Prediction: {prediction_label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        frame_window.image(img, channels="BGR")
        prediction_text.markdown(f"### Prediction: **{prediction_label or '-'}**")
