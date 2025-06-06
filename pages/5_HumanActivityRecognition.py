import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import streamlit as st
from PIL import Image

label = "Warmup...."
n_time_steps = 10
lm_list = []  # Global variable for storing landmarks

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model(
    "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/Human_Activity_Recognition/model.h5"
)

cap = cv2.VideoCapture(0)

# Streamlit button to control the process
start_button = st.button("Start")
stop_button = st.button("Stop")

stop_flag = False
detect_thread = None  # Variable to keep track of the detection thread


# Function for landmark processing
def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


# Function to draw landmarks on image
def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


# Function to draw the class label on image
def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(
        img,
        label,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )
    return img


# Function for making predictions
def detect(model, lm_list):
    global label
    if not lm_list:  # Check if lm_list is empty
        return "Waiting for data..."

    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)

    # Predict the class using the model
    results = model.predict(lm_list)

    if results is None or len(results) == 0:
        label = "Prediction Failed"
    else:
        if results[0][0] > 0.5:
            label = "SWING BODY"
        else:
            label = "SWING HAND"

    return label


# Streamlit app loop for video capture
def run_detection():
    global stop_flag, lm_list, detect_thread  # Declare lm_list and detect_thread as global
    warmup_frames = 60
    i = 0
    frame_placeholder = st.empty()  # Create a placeholder for the video frame

    while True:
        success, img = cap.read()
        if not success:
            st.write("Failed to capture image")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        i = i + 1

        if i > warmup_frames:
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)

                # Check if lm_list has enough elements before making a prediction
                if len(lm_list) >= n_time_steps:
                    if (
                        detect_thread is None or not detect_thread.is_alive()
                    ):  # Prevent new thread if previous is still running
                        detect_thread = threading.Thread(
                            target=detect,
                            args=(model, lm_list),
                        )
                        detect_thread.start()
                    lm_list = []  # Reset the list after prediction

                img = draw_landmark_on_image(mpDraw, results, img)

        img = draw_class_on_image(label, img)

        # Display image in Streamlit (in real-time)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame_placeholder.image(img, channels="RGB", use_container_width=True)

        # Stop the capture if stop button is pressed
        if stop_flag:
            stop_flag = True
            st.write("Detection stopped!")
            st.image("stop.jpg", caption="Detection Stopped", use_container_width=True)
            break

    cap.release()


# Button controls
if start_button:
    stop_flag = False
    st.write("Detection started!")
    run_detection()

elif stop_button:
    stop_flag = True
    st.write("Detection stopped!")
    st.image("stop.jpg", caption="Detection Stopped", use_container_width=True)
