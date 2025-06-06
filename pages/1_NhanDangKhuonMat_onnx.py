import numpy as np
import cv2 as cv
import joblib
import streamlit as st

# Set up Streamlit interface
st.subheader("Nhận dạng khuôn mặt")

# Initialize the video capture and stream display
cap = None
FRAME_WINDOW = st.image([])  # Initialize Streamlit image display

# Add button to start the webcam feed
if "start" not in st.session_state:
    st.session_state.start = False

if "stop" not in st.session_state:
    st.session_state.stop = False

# Button to start the webcam
press_start = st.button("Start")
if press_start:
    if not st.session_state.start:
        st.session_state.start = True
        cap = cv.VideoCapture(0)  # Start the webcam
        st.session_state.stop = False
    else:
        st.session_state.start = False
        if cap is not None:
            cap.release()  # Stop the webcam feed when "Stop" is pressed
            cap = None  # Set cap to None to ensure release happens only once

# Button to stop the webcam feed
press_stop = st.button("Stop")
if press_stop:
    if not st.session_state.stop:
        st.session_state.stop = True
        if cap is not None:
            cap.release()  # Stop the webcam feed
            cap = None  # Set cap to None to avoid errors
    else:
        st.session_state.stop = False
        cap = cv.VideoCapture(0)  # Start the webcam again

# Show stop image if webcam is stopped
if "frame_stop" not in st.session_state:
    stop_image = cv.imread(
        "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/NhanDangKhuonMat_onnx/stop.jpg"
    )
    st.session_state.frame_stop = stop_image

if st.session_state.stop:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels="BGR")
else:
    FRAME_WINDOW.image([])  # If not stopped, keep showing webcam feed

# Load model and face recognition functionality
svc = joblib.load(
    "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/NhanDangKhuonMat_onnx/model/svc.pkl"
)
mydict = ["DucThanh", "LeAnhTu"]  # Ensure these names match


# Function to visualize face detection results
def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            color = (0, 255, 0)  # Green bounding box for example

            if idx == 0:
                color = (255, 0, 0)  # Red for first person
            elif idx == 1:
                color = (0, 255, 0)  # Green for second person

            cv.rectangle(
                input,
                (coords[0], coords[1]),
                (coords[0] + coords[2], coords[1] + coords[3]),
                color,
                thickness,
            )

            # Recognition step (classify face)
            if faces[1] is not None:
                face_align = recognizer.alignCrop(input, face)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]

                center_x = coords[0] + coords[2] // 2
                center_y = coords[1] + coords[3] // 2
                cv.putText(
                    input,
                    result,
                    (center_x - 30, center_y + 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

    cv.putText(
        input,
        "FPS: {:.2f}".format(fps),
        (1, 16),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


# Main loop for webcam and face detection
if st.session_state.start and not st.session_state.stop:
    # Initialize OpenCV models
    detector = cv.FaceDetectorYN.create(
        "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/NhanDangKhuonMat_onnx/model/face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        0.9,
        0.3,
        5000,
    )
    recognizer = cv.FaceRecognizerSF.create(
        "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/NhanDangKhuonMat_onnx/model/face_recognition_sface_2021dec.onnx",
        "",
    )

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    tm = cv.TickMeter()

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            cv.putText(
                frame, result, (1, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Visualize face detection and recognition
        visualize(frame, faces, tm.getFPS())

        # Show live frame in Streamlit
        FRAME_WINDOW.image(frame, channels="BGR")
