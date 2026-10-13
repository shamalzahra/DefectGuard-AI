
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.predict import load_model, predict_image, DEVICE


st.set_page_config(
    page_title="DefectGuard AI — Inspection Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------
# DESIGN SYSTEM
# -------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #101321;
    --panel: #1a1e30;
    --border: #2b3047;
    --text: #f1f2fb;
    --muted: #9099b2;
    --purple: #8d70f6;
    --blue: #54a9f8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #101321;
    color: #f1f2fb;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1440px;
    padding: 1.2rem 2.4rem 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    letter-spacing: -0.04em;
    color: #f1f2fb !important;
}

h1 {
    font-size: 2rem !important;
}

h3 {
    font-size: 1.05rem !important;
}

p, label {
    color: #b8bfd1;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #1a1e30;
    border: 1px solid #2b3047;
    border-radius: 16px;
    overflow: hidden;
}

div[data-testid="stMetric"] {
    background: transparent;
}

div[data-testid="stMetricLabel"] {
    color: #929bb5;
    font-size: 0.85rem;
}

div[data-testid="stMetricValue"] {
    color: #f3f2ff;
    font-weight: 700;
}

div[data-testid="stFileUploader"] {
    background: #22263b;
    border-radius: 12px;
    padding: 6px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #22263b;
    border: 1px dashed #4b506b;
    border-radius: 10px;
}

div[data-testid="stFileUploaderDropzone"] button {
    background: #7561d5;
    color: white;
    border: 0;
}

div[data-testid="stRadio"] > div {
    gap: 16px;
}

div[data-testid="stRadio"] label {
    background: #202438;
    border-radius: 9px;
    padding: 7px 15px;
}

div[data-testid="stRadio"] label:has(input:checked) {
    background: #443875;
}

div[data-testid="stImage"] img {
    border-radius: 11px;
    object-fit: contain;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stPlotlyChart"] {
    border-radius: 12px;
}

.brand {
    color: #f5f5ff;
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.8px;
}

.brand-mark {
    color: #a18bff;
    font-size: 26px;
}

.page-title {
    font-family: 'Manrope', sans-serif;
    color: #f5f5ff;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin-top: 18px;
}

.page-subtitle {
    color: #8f99b2;
    font-size: 13px;
    margin-bottom: 16px;
}

.eyebrow {
    color: #a18bff;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.6px;
}

.status-dot {
    display: inline-block;
    height: 8px;
    width: 8px;
    border-radius: 50%;
    background: #49d2a0;
    margin-right: 7px;
}

.metric-big {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 32px;
    letter-spacing: -1.5px;
    color: #f5f4ff;
}

.muted {
    font-size: 12px;
    color: #939bb3;
}

.tag {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 7px;
    background: #342d56;
    color: #b5a5ff;
    font-size: 11px;
    font-weight: 700;
}

.result-name {
    color: #f7f4ff;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.action-text {
    color: #c1c7d8;
    font-size: 13px;
    line-height: 1.7;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


COLORS = [
    "#9479f5",
    "#56aaf4",
    "#6b79e8",
    "#e89270",
    "#67c9bd",
    "#b284ec",
]

RECOMMENDATIONS = {
    "crazing": "Inspect surface cracks and review processing conditions.",
    "inclusion": "Review material quality and possible contamination.",
    "patches": "Inspect surface treatment and affected regions.",
    "pitted_surface": "Inspect surface cavities and processing conditions.",
    "rolled-in_scale": "Review oxide scale removal and rolling conditions.",
    "scratches": "Inspect handling equipment and surface contact points.",
}


@st.cache_resource
def get_model():
    return load_model()


def display_name(name):
    return name.replace("_", " ").replace("-", " ").title()


def donut_chart(labels, values, colors=None, center_text=None):
    if colors is None:
        colors = COLORS

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.72,
                marker=dict(
                    colors=colors,
                    line=dict(color="#1a1e30", width=2),
                ),
                textinfo="none",
                hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
                sort=False,
            )
        ]
    )

    if center_text:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=center_text,
            showarrow=False,
            font=dict(
                size=23,
                color="#f5f4ff",
                family="Arial",
            ),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b5bdd4"),
        margin=dict(l=5, r=5, t=15, b=5),
        height=250,
        showlegend=False,
    )

    return fig


def bar_chart(labels, values):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color="#8870e6",
                cornerradius=5,
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=290,
        margin=dict(l=5, r=45, t=12, b=20),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="#292e43",
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#b9c2d7", size=11),
        ),
        font=dict(color="#b9c2d7"),
    )

    return fig


# -------------------------------------------
# TOP NAVIGATION
# -------------------------------------------

nav_left, nav_right = st.columns([4, 1.4])

with nav_left:
    st.markdown(
        '<div class="brand">'
        '<span class="brand-mark">◈</span> '
        'DefectGuard AI'
        '</div>',
        unsafe_allow_html=True,
    )

with nav_right:
    st.markdown(
        '<div style="text-align:right;padding-top:8px;">'
        '<span class="status-dot"></span>'
        '<span class="muted">Model online · '
        + html.escape(str(DEVICE)).upper()
        + '</span></div>',
        unsafe_allow_html=True,
    )

st.divider()

page = st.radio(
    "Navigation",
    ["Inspection", "Performance", "About"],
    horizontal=True,
    label_visibility="collapsed",
)

# -------------------------------------------
# INSPECTION WORKSPACE
# -------------------------------------------

if page == "Inspection":
    st.markdown(
        '<div class="page-title">Quality Inspection Console</div>'
        '<div class="page-subtitle">'
        'Industrial surface analysis / Defect classification'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Select a steel surface image",
        type=["jpg", "jpeg", "png"],
    )

    result = None
    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            model, classes = get_model()

            with st.spinner("Running image inspection..."):
                result = predict_image(image, model, classes)

        except (
            FileNotFoundError,
            RuntimeError,
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as error:
            st.error(
                f"Unable to complete inspection: {error}"
            )
            st.stop()

    # Upper panel
    col_image, col_analysis = st.columns(
        [1.2, 1], gap="medium"
    )

    with col_image:
        with st.container(border=True):
            st.markdown("### Inspection Image")

            if image is not None:
                st.image(
                    image,
                    use_container_width=True,
                )

                st.caption(
                    f"{uploaded_file.name} · "
                    f"{image.width} × {image.height} px"
                )
            else:
                st.info(
                    "Upload an image to display "
                    "the inspection preview."
                )

    with col_analysis:
        with st.container(border=True):
            st.markdown("### Defect Classification")

            if result:
                labels = [
                    display_name(name)
                    for name in result["scores"]
                ]

                values = [
                    score * 100
                    for score in result["scores"].values()
                ]

                st.plotly_chart(
                    donut_chart(
                        labels,
                        values,
                        center_text=(
                            f'{result["confidence"]:.1%}'
                        ),
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                st.markdown(
                    '<div class="muted">'
                    'Most likely defect category'
                    '</div>'
                    '<div class="result-name">'
                    + html.escape(display_name(result["class"]))
                    + '</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.plotly_chart(
                    donut_chart(
                        ["Awaiting Image"],
                        [100],
                        colors=["#343950"],
                        center_text="—",
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                st.caption("No analysis available yet.")

    st.write("")

    # Lower analytics
    bottom1, bottom2, bottom3 = st.columns(
        [1.15, 1, 0.9],
        gap="medium",
    )

    with bottom1:
        with st.container(border=True):
            st.markdown("### Classification Breakdown")

            if result:
                rows = sorted(
                    result["scores"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

                labels = [
                    display_name(name)
                    for name, _ in rows
                ]

                values = [
                    probability * 100
                    for _, probability in rows
                ]

                st.plotly_chart(
                    bar_chart(labels, values),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("Classification results will appear here.")

    with bottom2:
        with st.container(border=True):
            st.markdown("### Inspection Summary")

            if result:
                st.markdown(
                    '<div class="eyebrow">Detected category</div>'
                    '<div class="result-name">'
                    + html.escape(display_name(result["class"]))
                    + '</div>',
                    unsafe_allow_html=True,
                )

                st.write("")

                st.metric(
                    "Model confidence",
                    f'{result["confidence"]:.2%}',
                )

                if result["confidence"] < 0.70:
                    st.warning("Manual review recommended")
                else:
                    st.success("Classification completed")

                st.markdown("**Suggested action**")
                st.markdown(
                    '<div class="action-text">'
                    + html.escape(
                        RECOMMENDATIONS.get(
                            result["class"],
                            "Refer for manual inspection.",
                        )
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Waiting for inspection data.")

    with bottom3:
        with st.container(border=True):
            st.markdown("### Model Performance")

            st.plotly_chart(
                donut_chart(
                    ["Correct", "Incorrect"],
                    [95.28, 4.72],
                    colors=["#9278f4", "#30364d"],
                    center_text="95.28%",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.markdown(
                '<div style="text-align:center">'
                '<div class="muted">Held-out test accuracy</div>'
                '<div class="metric-big">95.28%</div>'
                '<div class="muted">343 / 360 images correct</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.caption(
        "Research prototype · Classification only. "
        "No live camera feed or factory integration. "
        "Predictions require human verification."
    )


# -------------------------------------------
# PERFORMANCE PAGE
# -------------------------------------------

elif page == "Performance":
    st.markdown(
        '<div class="page-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Evaluation results from 360 held-out NEU images."
    )

    a, b, c = st.columns(3)

    a.metric("Test Accuracy", "95.28%")
    b.metric("Validation Accuracy", "97.92%")
    c.metric("Correct Predictions", "343 / 360")

    st.write("")

    report = pd.DataFrame({
        "Defect Category": [
            "Crazing",
            "Inclusion",
            "Patches",
            "Pitted Surface",
            "Rolled-in Scale",
            "Scratches",
        ],
        "Precision": [
            0.95, 1.00, 0.98, 0.98, 1.00, 0.83
        ],
        "Recall": [
            1.00, 0.78, 0.98, 1.00, 0.95, 1.00
        ],
        "F1 Score": [
            0.98, 0.88, 0.98, 0.99, 0.97, 0.91
        ],
    })

    with st.container(border=True):
        st.markdown("### Per-Class Performance")
        st.dataframe(
            report,
            hide_index=True,
            use_container_width=True,
        )

    st.warning(
        "Inclusion recall is 78%. The model needs "
        "further validation before industrial use."
    )

    matrix_path = (
        Path(__file__).resolve().parent
        / "results"
        / "confusion_matrix.png"
    )

    if matrix_path.exists():
        with st.container(border=True):
            st.markdown("### Confusion Matrix")
            st.image(
                str(matrix_path),
                use_container_width=True,
            )


# -------------------------------------------
# ABOUT PAGE
# -------------------------------------------

else:
    st.markdown(
        '<div class="page-title">About DefectGuard AI</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("""
        ### Industrial Quality Inspection

        DefectGuard AI is an AI-assisted steel-surface
        defect classification prototype developed for
        **AMD Developer Hackathon: ACT III**.

        **Track:** Intelligent Industry

        **Model:** ResNet18 with transfer learning

        **Framework:** PyTorch

        **Interface:** Streamlit and Plotly

        **Dataset:** NEU Surface Defect Database

        **Current execution:** CPU

        **Planned:** AMD GPU inference and benchmarking

        ### Known limitations

        The system classifies six defect categories.
        It has not been trained to identify defect-free
        surfaces or validated in a production factory.

        AI suggestions are advisory and require
        human verification.
        """)

st.divider()

st.caption(
    "DefectGuard AI · Industrial Intelligence Console"
    " · AMD Developer Hackathon ACT III · 2026"
)
