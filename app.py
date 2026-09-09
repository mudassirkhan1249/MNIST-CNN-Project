import streamlit as st
import numpy as np
import uuid
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="MNIST CNN Digit Recognizer",
    page_icon="✍️",
    layout="wide",
)

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "digit_canvas_0"

st.markdown("""
<style>
.hero {
    padding: 1.2rem 1.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(14,165,233,.12));
    border: 1px solid rgba(128,128,128,.22);
    margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2.25rem; }
.hero p { margin: .35rem 0 0; opacity: .78; }
.card {
    padding: 1.15rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 16px;
    background: rgba(128,128,128,.06);
}
.pred {
    text-align: center;
    padding: 1.4rem;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.22);
}
.pred .digit { font-size: 5rem; font-weight: 800; line-height: 1; }
.pred .label { opacity: .7; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cnn():
    return load_model("cnn_model.h5")

def preprocess_canvas(pil_img):
    """Convert the drawing canvas into the MNIST 28x28 format."""
    img = pil_img.convert("L")
    arr = np.array(img)

    # Canvas is black background with white ink.
    # If a white-background image is supplied, invert it automatically.
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox is None:
        return np.zeros((1, 28, 28, 1), dtype=np.float32), img

    # Crop to the actual digit.
    digit = img.crop(bbox)

    # Keep a margin and resize while preserving aspect ratio.
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    canvas.paste(digit, (x, y))

    processed = np.array(canvas, dtype=np.float32) / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    return processed, canvas

st.markdown("""
<div class="hero">
    <h1>✍️ MNIST CNN Digit Recognizer</h1>
    <p>Draw a digit with your mouse or trackpad, then let the CNN recognize it.</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_cnn()
except Exception as e:
    st.error("Model load nahi ho raha. Make sure `cnn_model.h5` app.py ke same folder mein hai.")
    st.exception(e)
    st.stop()

# Streamlit drawable canvas component
from streamlit_drawable_canvas import st_canvas

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("### 🎨 Draw your digit")
    st.caption("Black canvas par white brush se 0–9 mein se koi bhi digit draw karein.")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=420,
        height=420,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        return_image_data=True,
    )

    

    predict_clicked = st.button(
        "🔮 Predict Digit",
        type="primary",
        use_container_width=True
    )

    clear_clicked = st.button(
        "🧹 Clear Canvas",
        use_container_width=True
    )

    if clear_clicked:
        st.session_state.canvas_key = f"digit_canvas_{uuid.uuid4()}"
        st.rerun()

with right:
    st.markdown("### 🤖 Prediction")

    if predict_clicked:
        if canvas_result.image_data is None:
            st.warning("Pehle canvas par digit draw karein.")
        else:
            image = Image.fromarray(canvas_result.image_data.astype("uint8"))
            processed, preview = preprocess_canvas(image)

            with st.spinner("CNN is thinking..."):
                probabilities = model.predict(processed, verbose=0)[0]

            predicted_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_digit]) * 100

            st.markdown(f"""
            <div class="pred">
                <div class="label">Predicted Digit</div>
                <div class="digit">{predicted_digit}</div>
                <div class="label">Confidence: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            top3 = np.argsort(probabilities)[::-1][:3]
            c1, c2, c3 = st.columns(3)
            for col, idx in zip((c1, c2, c3), top3):
                col.metric(
                    f"Digit {idx}",
                    f"{probabilities[idx] * 100:.2f}%"
                )

            st.markdown("### 📊 Class probabilities")
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.bar(range(10), probabilities * 100)
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            with st.expander("🔍 View 28×28 processed image"):
                st.image(
                    preview,
                    caption="Image sent to the CNN",
                    width=220
                )
    else:
        st.info("Draw a digit on the left and click **Predict Digit**.")

st.divider()

st.markdown("""
<div style="text-align:center; opacity:.65;">
    CNN architecture: Conv2D → MaxPooling → Flatten → Dense → Softmax
    <br>
    Input: 28×28 grayscale image • Classes: 0–9
</div>
""", unsafe_allow_html=True)
