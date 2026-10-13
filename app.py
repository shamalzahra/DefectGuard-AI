
import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.predict import load_model, predict_image, DEVICE

st.set_page_config(
    page_title="DefectGuard AI",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 DefectGuard AI")
st.subheader("Intelligent Industrial Quality Inspection")

st.caption(
    "AMD Developer Hackathon: ACT III | "
    "Track 1 — Intelligent Industry"
)

st.divider()

RECOMMENDATIONS = {
    "crazing": (
        "Flag for human inspection of surface cracking "
        "and review relevant processing conditions."
    ),
    "inclusion": (
        "Flag for inspection of possible foreign material "
        "and review material quality controls."
    ),
    "patches": (
        "Inspect the affected surface and review "
        "surface treatment conditions."
    ),
    "pitted_surface": (
        "Inspect surface cavities and investigate "
        "possible corrosion or processing issues."
    ),
    "rolled-in_scale": (
        "Inspect for embedded oxide scale and review "
        "surface cleaning and rolling processes."
    ),
    "scratches": (
        "Inspect surface contact points, handling "
        "equipment, and possible abrasion sources."
    ),
}


@st.cache_resource
def get_model():
    return load_model()


with st.sidebar:
    st.header("System Information")
    st.write("Model: ResNet18")
    st.write("Dataset: NEU Surface Defect")
    st.write(f"Device: {DEVICE}")
    st.metric("Held-out Test Accuracy", "95.28%")
    st.caption("360 NEU test images; not field-validated.")


st.header("Surface Defect Classification")

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        model, classes = get_model()
        result = predict_image(image, model, classes)

    except (UnidentifiedImageError, OSError) as error:
        st.error(f"Unable to read the image: {error}")
        st.stop()

    except (FileNotFoundError, RuntimeError) as error:
        st.error(f"Model loading or prediction failed: {error}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Surface Image",
            use_container_width=True,
        )

    with col2:
        st.subheader("AI Inspection Result")

        defect_name = (
            result["class"].replace("_", " ").title()
        )

        confidence = result["confidence"]

        st.metric(
            "Predicted Defect Type",
            defect_name,
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2%}",
        )

        if confidence < 0.70:
            st.warning(
                "Low model confidence. "
                "Manual review is recommended."
            )
        else:
            st.info(
                "Prediction generated. "
                "Human verification is still required."
            )

        st.write("**Suggested Inspection Action**")
        st.write(
            RECOMMENDATIONS.get(
                result["class"],
                "Refer this image for manual inspection.",
            )
        )

    st.divider()

    st.subheader("Class Probability Distribution")

    probability_data = pd.DataFrame(
        {
            "Defect Type": [
                name.replace("_", " ").title()
                for name in result["scores"]
            ],
            "Probability": [
                value * 100
                for value in result["scores"].values()
            ],
        }
    )

    st.bar_chart(
        probability_data.set_index("Defect Type")
    )

    st.caption(
        "Probabilities are model outputs, not calibrated "
        "reliability guarantees. This prototype classifies "
        "six known defect categories and cannot establish "
        "that a surface is defect-free."
    )

else:
    st.info("Upload an image to start defect classification.")

st.divider()

st.caption(
    "DefectGuard AI | Research Prototype | "
    "AMD Developer Hackathon ACT III"
)
