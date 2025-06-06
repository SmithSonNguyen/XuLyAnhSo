import os
from PIL import Image
from ultralytics import YOLO
import supervision as sv
import streamlit as st

# Kiểm tra cài đặt thư viện
import ultralytics

# Run the necessary checks for the ultralytics library
ultralytics.checks()

# Load the pre-trained YOLOv8 model
model = YOLO(
    "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/Toi_2_4_6_TraiCay640x640_yolov8/train_yolov8n/runs/detect/train/weights/best.pt"
)

# Set page title
st.title("Dự Đoán Trái Cây với YOLOv8")

# File uploader widget with description
uploaded_file = st.file_uploader("Tải ảnh lên để dự đoán", type=["png", "jpg", "jpeg"])

# If a file is uploaded, process it
if uploaded_file is not None:
    try:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image for the user
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

        # Progress spinner while making predictions
        with st.spinner("Đang dự đoán..."):
            # Perform prediction with confidence threshold of 0.25
            results = model.predict(image, conf=0.25)[0]

            # Convert the results into the 'supervision' format for annotations
            detections = sv.Detections.from_ultralytics(results)

            # Prepare the annotators
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

            # Make a copy of the image to annotate
            annotated_image = image.copy()
            annotated_image = box_annotator.annotate(
                annotated_image, detections=detections
            )
            annotated_image = label_annotator.annotate(
                annotated_image, detections=detections
            )

            # Display the annotated image with results
            st.image(
                annotated_image,
                caption="Ảnh dự đoán với YOLOv8",
                use_container_width=True,
            )

    except Exception as e:
        # Handle errors gracefully with a user-friendly message
        st.error(f"Đã xảy ra lỗi: {e}")

else:
    # Info message when no file is uploaded
    st.info("Vui lòng tải lên một bức ảnh để dự đoán.")
