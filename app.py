import streamlit as st


# Set up the page configuration
st.set_page_config(page_title="Student Portfolio", page_icon="📚")


# Main header for the app
st.title("Welcome to Our Streamlit App! 👋")

# Initialize session state for pages to ensure clearing of old page data
if "current_page" not in st.session_state:
    st.session_state.current_page = None

# Sidebar page navigation
pages = {
    "GioiThieu": "pages/0_GioiThieuNhom.py",
    "NhanDangKhuonMat": "pages/1_NhanDangKhuonMat_onnx.py",
    "NhanDienTraiCay": "pages/2_NhanDienTraiCay_yolov8.py",
    "Xu Ly Anh So": "pages/3_xulyanhso.py",
    "NhanDangChuSo": "pages/4_NhanDangChuSo.py",
    "HumanActivityRecognition": "pages/5_HumanActivityRecognition.py",
}

# Page selection
page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# If the selected page is different from the current page, reset the content
if page != st.session_state.current_page:
    st.session_state.current_page = page
    # You can clear session state variables associated with the old page here if needed
    # For example, clearing variables associated with previous page's data:
    if "previous_data" in st.session_state:
        del st.session_state.previous_data

# Load the selected page with utf-8 encoding to avoid UnicodeDecodeError
if page in pages:
    try:
        with open(pages[page], encoding="utf-8") as f:
            exec(f.read())
    except Exception as e:
        st.error(f"Error loading the selected page: {e}")
