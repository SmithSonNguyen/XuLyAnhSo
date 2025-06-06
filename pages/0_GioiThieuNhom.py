import streamlit as st
from PIL import Image
import base64
from io import BytesIO


# Function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Background image or color
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


background_image_path = (
    "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/image_main/bg.png"
)
background_base64 = get_image_base64(background_image_path)

st.markdown(
    f"""
    <style>
    body {{
        background-color: #f1f6f9;
        background-image: url('data:image/png;base64,{background_base64}');
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Main header with styling
st.markdown(
    """
    <style>
    h1 {
        font-size: 3em;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to load images from a local path
def load_image(image_path):
    img = Image.open(image_path)
    return img


# Developer Section with images and cards
st.markdown(
    """
    <style>
    .developer-card {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
    }
    .developer-card h3 {
        color: #0057d9;
    }
    .developer-card p {
        color: #666;
    }
    .developer-card img {
        border-radius: 50%;
        width: 80px;
        height: 80px;
        margin-right: 20px;
        float: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load and display the developer images using st.image
img_thanh = load_image(
    "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/image_main/ducthanh.png"
)
img_tu = load_image(
    "D:/CacmonhocdaihocSPKT/Ki6/XuLyAnhSo_Duc/cuoiki/image_main/anhtus.jpg"
)

st.markdown(
    """
    <div class="developer-card">
        <img src="data:image/png;base64,{}" alt="Nguyễn Đức Thành">
        <h3>Nguyễn Đức Thành</h3>
        <p><strong>MSSV:</strong> 22110416</p>
        <p>Passionate about <strong>AI</strong> and <strong>data analysis</strong>.</p>
        <p>Aiming to bring cutting-edge <strong>machine learning models</strong> to life.</p>
    </div>
    
    <div class="developer-card">
        <img src="data:image/png;base64,{}" alt="Lê Anh Tú">
        <h3>Lê Anh Tú</h3>
        <p><strong>MSSV:</strong> 22110453</p>
        <p>Enthusiastic about <strong>deep learning</strong> and <strong>automation</strong>.</p>
        <p>Always exploring new ways to optimize <strong>data-driven systems</strong>.</p>
    </div>
    """.format(
        image_to_base64(img_thanh), image_to_base64(img_tu)
    ),
    unsafe_allow_html=True,
)

# About Section with background
st.markdown(
    """
    <style>
    .about-section {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .about-section h2 {
        color: #333;
    }
    .about-section img {
        width: 100%;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="about-section">
        <h2>What is this app about?</h2>
        <p>This app is a demo built with <strong>Streamlit</strong> to showcase the power of creating interactive data science applications. This project is developed by two students who are passionate about machine learning and data science.</p>
        <img src="https://anhphuongit.com/DATA/Images/cam-bien-thi-giac-thi-giac-may-tinh-xu-ly-anh-la-gi.jpg" alt="Machine Learning Image">
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with navigation
st.sidebar.title("Navigation")
st.sidebar.success("Select a demo above.")

# Footer with credits and social media
st.markdown(
    """
    <style>
    .footer {
        font-size: 0.9em;
        text-align: center;
        padding: 20px;
        color: #333;
        background-color: #eeeeee;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="footer">
        Built by Nguyễn Đức Thành and Lê Anh Tú.   
        Follow us on [GitHub](https://github.com) for more updates!
    </div>
    """,
    unsafe_allow_html=True,
)
