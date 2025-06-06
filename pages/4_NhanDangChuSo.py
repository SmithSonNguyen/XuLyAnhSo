import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model (make sure the path is correct)
model = load_model(
    "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/Nhan_dang_chu_so/mnist_cnn_augmented.h5"
)

# ============================
# 1. Create Streamlit Interface
# ============================
st.title("Nhận diện chữ số viết tay với CNN 🖋️")

# Description
st.markdown(
    """
    Tải ảnh chữ số của bạn lên và mô hình sẽ nhận diện ngay lập tức. Chỉ cần tải ảnh có chữ số rõ ràng, mô hình CNN sẽ làm việc.
    Hãy thử tải lên ảnh của bạn!
    """,
    unsafe_allow_html=True,
)

# Upload image from the user
image_file = st.file_uploader(
    "📥 Chọn ảnh chữ số viết tay", type=["jpg", "png", "jpeg"]
)

# ============================
# 2. Predict button to trigger prediction
# ============================
if st.button("Nhận diện chữ số"):
    if image_file is not None:
        # Load the uploaded image from the Streamlit uploader
        image = Image.open(image_file)
        image = np.array(image)

        if image is None:
            st.error("Không thể đọc ảnh!")
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

            # Find contours of digits
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(
                contours, key=lambda c: cv2.boundingRect(c)[0]
            )  # Sort from left to right

            # Process each digit
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = binary[y : y + h, x : x + w]

                # Padding and resizing the digit
                roi = np.pad(
                    roi, ((20, 20), (20, 20)), mode="constant", constant_values=0
                )
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = roi.astype("float32") / 255.0
                roi = roi.reshape(1, 28, 28, 1)

                # Predict with the CNN model
                prediction = model.predict(roi)
                digit = np.argmax(prediction)

                # Display the result on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    str(digit),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2,
                    (0, 255, 255),
                    2,
                )

            # Convert the result back to PIL for Streamlit
            result_image = Image.fromarray(image)

            # Display the result image in Streamlit
            st.image(
                result_image, caption="Kết quả nhận diện chữ số", use_column_width=True
            )

    else:
        st.error("Vui lòng tải lên ảnh chữ số để nhận diện!")
