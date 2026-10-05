
import base64
import io
import json
from datetime import datetime

import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://localhost:8000/predict"
MAX_UPLOAD_MB = 10


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GI Endoscopy Image Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GENERAL
       ========================= */

    .stApp {
        background-color: #f5f8fc;
    }

    .block-container {
        padding-top: 0;
        padding-bottom: 2rem;
        max-width: 1400px;
    }


    /* =========================
       HEADER
       ========================= */

    .top-header {
        background: linear-gradient(
            135deg,
            #162d48 0%,
            #233f60 100%
        );

        padding: 28px 35px;
        margin-left: -5rem;
        margin-right: -5rem;
        margin-bottom: 28px;

        color: white;

        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 30px;
    }

    .header-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }

    .header-subtitle {
        font-size: 18px;
        margin-top: 8px;
        color: #d8e3ef;
    }

    .research-badge {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 14px 20px;
        min-width: 300px;
        font-size: 14px;
        color: white;
    }

    .research-title {
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 4px;
    }


    /* =========================
       CARDS
       ========================= */

    .main-card {
        background: white;
        border: 1px solid #e1e8f0;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 2px 10px rgba(30, 60, 90, 0.05);
        min-height: 500px;
    }

    .card-title {
        font-size: 23px;
        font-weight: 700;
        color: #173250;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #173250;
        margin-top: 10px;
        margin-bottom: 15px;
    }


    /* =========================
       PREDICTION
       ========================= */

    .prediction-card {
        background: linear-gradient(
            135deg,
            #effaf5,
            #f5fcf8
        );

        border: 1px solid #ccebdc;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .prediction-label {
        color: #18794e;
        font-size: 16px;
        font-weight: 600;
    }

    .prediction-class {
        color: #269b67;
        font-size: 38px;
        font-weight: 750;
        margin-top: 3px;
        margin-bottom: 18px;
        text-transform: capitalize;
    }

    .confidence-label {
        color: #344e68;
        font-size: 17px;
        font-weight: 600;
    }

    .confidence-value {
        color: #269b67;
        font-size: 34px;
        font-weight: 750;
    }


    /* =========================
       PROBABILITY BARS
       ========================= */

    .prob-row {
        margin-bottom: 15px;
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        color: #243b53;
        font-size: 15px;
    }

    .prob-name {
        font-weight: 600;
    }

    .prob-value {
        font-weight: 600;
    }

    .prob-background {
        width: 100%;
        height: 10px;
        background: #e6edf5;
        border-radius: 8px;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        background: linear-gradient(
            90deg,
            #3478e5,
            #4b91f1
        );
        border-radius: 8px;
    }

    .prob-fill-high {
        height: 100%;
        background: linear-gradient(
            90deg,
            #28a76b,
            #48bd88
        );
        border-radius: 8px;
    }


    /* =========================
       INFORMATION BOX
       ========================= */

    .info-box {
        background: #edf4ff;
        border: 1px solid #d5e4fa;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 20px;
        color: #36516f;
    }

    .info-title {
        color: #3478e5;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 7px;
    }


    /* =========================
       GRAD-CAM
       ========================= */

    .gradcam-card {
        background: white;
        border: 1px solid #e1e8f0;
        border-radius: 14px;
        padding: 25px;
        margin-top: 25px;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #718096;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #e1e8f0;
    }


    /* =========================
       STREAMLIT UI
       ========================= */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-header">

        <div>
            <div class="header-title">
                GI Endoscopy Image Classification
            </div>

            <div class="header-subtitle">
                AI-powered support for gastrointestinal image analysis
            </div>
        </div>

        <div class="research-badge">
            <div class="research-title">
                ℹ️ &nbsp; For educational and research use only
            </div>

            <div>
                Not a diagnostic tool
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an endoscopic image",
    type=["jpg", "jpeg", "png"],
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    # Remove previous result if no image is selected.
    st.session_state.pop("result", None)
    st.session_state.pop("uploaded_filename", None)

    st.markdown(
        """
        <div style="
            background:white;
            border:1px solid #e1e8f0;
            border-radius:14px;
            padding:55px;
            text-align:center;
            margin-top:20px;
        ">

            <div style="font-size:55px;">
                🔬
            </div>

            <h2 style="color:#173250;">
                Upload an Endoscopy Image
            </h2>

            <p style="
                color:#718096;
                font-size:17px;
            ">
                Upload a JPG, JPEG or PNG image to analyze it
                using the fine-tuned ResNet50 model.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# RESET RESULT WHEN A NEW FILE IS SELECTED
# ============================================================

current_filename = uploaded_file.name

if (
    "uploaded_filename" not in st.session_state
    or st.session_state["uploaded_filename"] != current_filename
):

    st.session_state.pop("result", None)
    st.session_state["uploaded_filename"] = current_filename


# ============================================================
# FILE SIZE
# ============================================================

file_size_mb = uploaded_file.size / (1024 * 1024)

if file_size_mb > MAX_UPLOAD_MB:

    st.error(
        f"File is too large. Maximum allowed size is "
        f"{MAX_UPLOAD_MB} MB."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(uploaded_file).convert("RGB")

except Exception:

    st.error("Unable to read the uploaded image.")
    st.stop()


# ============================================================
# MAIN COLUMNS
# ============================================================

left_col, right_col = st.columns(
    [1.08, 0.92],
    gap="large",
)


# ============================================================
# LEFT COLUMN — IMAGE
# ============================================================

with left_col:

    st.markdown(
        """
        <div class="main-card">

            <div class="card-title">
                🖼️ &nbsp; Uploaded Image
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(
        image,
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div style="
            color:#718096;
            font-size:14px;
            margin-top:10px;
        ">

            <b>File:</b> {uploaded_file.name}<br>

            <b>Resolution:</b>
            {image.width} × {image.height}<br>

            <b>Size:</b>
            {file_size_mb:.2f} MB

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# RIGHT COLUMN — PREDICTION
# ============================================================

with right_col:

    st.markdown(
        """
        <div class="main-card">

            <div class="card-title">
                📄 &nbsp; Prediction Result
            </div>
        """,
        unsafe_allow_html=True,
    )

    analyze = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True,
    )


    # ========================================================
    # CALL FASTAPI
    # ========================================================

    if analyze:

        with st.spinner(
            "Analyzing image with the ResNet50 model..."
        ):

            try:

                image_bytes = io.BytesIO()

                image.save(
                    image_bytes,
                    format="JPEG",
                )

                image_bytes.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        image_bytes,
                        "image/jpeg",
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120,
                )


                # --------------------------------------------
                # HTTP ERROR
                # --------------------------------------------

                if response.status_code != 200:

                    st.error(
                        f"FastAPI returned HTTP "
                        f"{response.status_code}"
                    )

                    try:
                        st.json(response.json())
                    except Exception:
                        st.code(response.text)

                    st.stop()


                # --------------------------------------------
                # JSON RESPONSE
                # --------------------------------------------

                try:

                    result = response.json()

                except ValueError:

                    st.error(
                        "FastAPI did not return valid JSON."
                    )

                    st.code(response.text)

                    st.stop()


                st.session_state["result"] = result


            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Unable to connect to FastAPI."
                )

                st.info(
                    "Make sure your Docker container is running "
                    "and that FastAPI is available at "
                    "http://localhost:8000."
                )

                st.stop()


            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The request to FastAPI timed out."
                )

                st.stop()


            except requests.exceptions.RequestException as error:

                st.error(
                    f"❌ Request error: {error}"
                )

                st.stop()


            except Exception as error:

                st.error(
                    f"❌ Unexpected error: {error}"
                )

                st.stop()


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if "result" in st.session_state:

        result = st.session_state["result"]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = result.get(
            "prediction",
            result.get("class", "Unknown"),
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        try:

            confidence = float(
                result.get("confidence", 0)
            )

        except (TypeError, ValueError):

            confidence = 0.0


        # ----------------------------------------------------
        # NORMALIZE CONFIDENCE
        # ----------------------------------------------------

        if confidence > 1:
            confidence = confidence / 100

        confidence = max(
            0.0,
            min(1.0, confidence),
        )


        # ----------------------------------------------------
        # DISPLAY NAME
        # ----------------------------------------------------

        display_prediction = str(
            prediction
        ).replace(
            "-",
            " ",
        ).title()


        # ----------------------------------------------------
        # PREDICTION CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    ✓ &nbsp; Predicted Class
                </div>

                <div class="prediction-class">
                    {display_prediction}
                </div>

                <div class="confidence-label">
                    Confidence
                </div>

                <div class="confidence-value">
                    {confidence * 100:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probabilities = result.get(
            "probabilities",
            result.get(
                "class_probabilities",
                {},
            ),
        )


        if isinstance(probabilities, dict) and probabilities:

            st.markdown(
                """
                <div class="section-title">
                    📊 &nbsp; Class Probabilities
                </div>
                """,
                unsafe_allow_html=True,
            )


            # Sort highest probability first.

            sorted_probabilities = sorted(
                probabilities.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )


            for class_name, probability in sorted_probabilities:

                try:
                    probability = float(probability)
                except (TypeError, ValueError):
                    continue


                # Handle values returned as percentages.

                if probability > 1:
                    probability = probability / 100


                probability = max(
                    0.0,
                    min(1.0, probability),
                )


                percentage = probability * 100


                display_name = str(
                    class_name
                ).replace(
                    "-",
                    " ",
                ).title()


                is_prediction = (
                    str(class_name).lower()
                    == str(prediction).lower()
                )


                fill_class = (
                    "prob-fill-high"
                    if is_prediction
                    else "prob-fill"
                )


                st.markdown(
                    f"""
                    <div class="prob-row">

                        <div class="prob-header">

                            <span class="prob-name">
                                {display_name}
                            </span>

                            <span class="prob-value">
                                {percentage:.1f}%
                            </span>

                        </div>

                        <div class="prob-background">

                            <div
                                class="{fill_class}"
                                style="width:{percentage:.1f}%"
                            ></div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # ----------------------------------------------------
        # NOTE
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="info-box">

                <div class="info-title">
                    ℹ️ &nbsp; Note
                </div>

                This result is generated by an AI model
                for educational and research purposes only.
                It is not a medical diagnosis.

            </div>
            """,
            unsafe_allow_html=True,
        )


    else:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:70px 20px;
                color:#718096;
            ">

                <div style="font-size:48px;">
                    🧠
                </div>

                <h3 style="color:#344e68;">
                    Ready for analysis
                </h3>

                <p>
                    Click <b>Analyze Image</b> to send the image
                    to the FastAPI model.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# GRAD-CAM
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    gradcam = result.get("gradcam")


    st.markdown(
        """
        <div class="gradcam-card">

            <div class="section-title">
                🔥 &nbsp; Grad-CAM Explainability
            </div>

            <p style="color:#718096;">
                Grad-CAM highlights the regions of the image
                that contributed most strongly to the model's
                prediction.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    if gradcam:

        # ====================================================
        # SINGLE IMAGE
        # ====================================================

        if isinstance(gradcam, str):

            try:

                if gradcam.startswith("data:image"):

                    _, encoded = gradcam.split(
                        ",",
                        1,
                    )

                    image_data = base64.b64decode(
                        encoded
                    )

                    gradcam_image = Image.open(
                        io.BytesIO(image_data)
                    )

                    st.image(
                        gradcam_image,
                        caption="Grad-CAM visualization",
                        use_container_width=True,
                    )

                else:

                    st.image(
                        gradcam,
                        caption="Grad-CAM visualization",
                        use_container_width=True,
                    )

            except Exception as error:

                st.warning(
                    f"Grad-CAM could not be displayed: {error}"
                )


        # ====================================================
        # MULTIPLE IMAGES
        # ====================================================

        elif isinstance(gradcam, dict):

            for class_name, heatmap in gradcam.items():

                display_name = str(
                    class_name
                ).replace(
                    "-",
                    " ",
                ).title()


                st.markdown(
                    f"### {display_name}"
                )


                try:

                    if isinstance(heatmap, str) and heatmap.startswith(
                        "data:image"
                    ):

                        _, encoded = heatmap.split(
                            ",",
                            1,
                        )

                        image_data = base64.b64decode(
                            encoded
                        )

                        heatmap_image = Image.open(
                            io.BytesIO(image_data)
                        )

                        st.image(
                            heatmap_image,
                            caption=f"Grad-CAM — {display_name}",
                            use_container_width=True,
                        )

                    else:

                        st.image(
                            heatmap,
                            caption=f"Grad-CAM — {display_name}",
                            use_container_width=True,
                        )

                except Exception as error:

                    st.warning(
                        f"Could not display Grad-CAM for "
                        f"{display_name}: {error}"
                    )


    else:

        st.info(
            "The API did not return a Grad-CAM visualization."
        )


# ============================================================
# BOTTOM ACTIONS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

button_col1, button_col2 = st.columns(2)


# ============================================================
# UPLOAD ANOTHER IMAGE
# ============================================================

with button_col1:

    if st.button(
        "🔄 Upload Another Image",
        use_container_width=True,
    ):

        st.session_state.pop(
            "result",
            None,
        )

        st.session_state.pop(
            "uploaded_filename",
            None,
        )

        st.rerun()


# ============================================================
# DOWNLOAD RESULT
# ============================================================

with button_col2:

    if "result" in st.session_state:

        download_data = json.dumps(
            st.session_state["result"],
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            label="⬇️ Download Result",
            data=download_data,
            file_name=(
                "prediction_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ),
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        HyperKvasir AI Classification System
        • Fine-tuned ResNet50
        • Educational & Research Use Only

    </div>
    """,
    unsafe_allow_html=True,
)
