import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from io import BytesIO
import Chapter3 as c3


class StreamlitApp:
    def __init__(self):
        self.model = YOLO(
            "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/ThucHanhXuLyAnh/yolov8n_trai_cay.pt",
            task="detect",
        )

    def display_title(self):
        st.title("Xử lý ảnh số")

    def upload_image(self, color_image=True):
        """
        Uploads image (either grayscale or color depending on 'color_image' flag).
        """
        uploaded_file = st.file_uploader(
            "Chọn hình ảnh", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
        )
        if uploaded_file is not None:
            # Convert the uploaded image into a format that OpenCV can work with
            image_bytes = uploaded_file.read()
            img = np.array(bytearray(image_bytes), dtype=np.uint8)
            if color_image:
                img = cv2.imdecode(
                    img, cv2.IMREAD_COLOR
                )  # OpenCV reads the image as color
            else:
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            return img
        return None

    def display_image(self, img, title="Image"):
        """
        Displays image on Streamlit app.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=title, use_container_width=True)

    def apply_c3_operations(self, img, operation):
        """
        Applies various image operations from Chapter 3.
        """
        operations = {
            "Negative": c3.Negative,
            "Negative Color": c3.NegativeColor,
            "Logarit": c3.Logarit,
            "Power": c3.Power,
            "Piecewise Line": c3.PiecewiseLine,
            "Histogram": c3.Histogram,
            "Hist Equal": cv2.equalizeHist,
            "Hist Equal Color": c3.HistEqualColor,
            "Local Hist": c3.LocalHist,
            "Hist Stat": c3.HistStat,
            "Smooth Box": lambda img: cv2.boxFilter(img, cv2.CV_8UC1, (21, 21)),
            "Smooth Gauss": lambda img: cv2.GaussianBlur(img, (43, 43), 7.0),
            "Median Filter": lambda img: cv2.medianBlur(img, 5),
            "Create Impulse Noise": c3.CreateImpulseNoise,
            "Sharp": c3.Sharp,
        }

        return operations.get(operation, lambda img: img)(img)

    def run(self):
        self.display_title()

        # Select image type (color or grayscale)
        image_type = st.radio("Select Image Type", ("Color Image", "Grayscale Image"))

        # Upload Image
        is_color = image_type == "Color Image"
        img = self.upload_image(color_image=is_color)
        if img is not None:
            # Show the uploaded image
            self.display_image(img, "Uploaded Image")

            # Operations from Chapter 3
            operation = st.selectbox(
                "Choose an image operation",
                [
                    "Negative",
                    "Negative Color",
                    "Logarit",
                    "Power",
                    "Piecewise Line",
                    "Histogram",
                    "Hist Equal",
                    "Hist Equal Color",
                    "Local Hist",
                    "Hist Stat",
                    "Smooth Box",
                    "Smooth Gauss",
                    "Median Filter",
                    "Create Impulse Noise",
                    "Sharp",
                ],
            )
            if st.button(f"Apply {operation}"):
                imgout = self.apply_c3_operations(img, operation)
                self.display_image(imgout, f"Result of {operation}")


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
