
import streamlit as st
from PIL import Image

# Configure application
st.set_page_config(
    page_title="DefectGuard AI",
    page_icon="🏭",
    layout="wide"
)

# Header
st.title("🏭 DefectGuard AI")
st.subheader("Intelligent Industrial Quality Inspection")

st.write(
    "AMD Developer Hackathon: ACT III | "
    "Track 1 — Intelligent Industry"
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("About DefectGuard AI")
    st.write(
        "An AI-powered quality inspection system "
        "for industrial surface defects."
    )
    st.info("Version 0.1 — Initial Prototype")

# Image upload
st.header("Industrial Image Inspection")

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:
        st.subheader("Inspection Results")

        st.success("Image uploaded successfully!")

        st.warning(
            "AI model not yet connected. "
            "Defect classification will be "
            "available after model training."
        )

else:
    st.info("Please upload an image to begin inspection.")

st.divider()

st.caption(
    "DefectGuard AI | AMD Developer Hackathon ACT III"
)
