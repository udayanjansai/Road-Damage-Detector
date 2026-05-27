import os
import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Damage Detector",
    page_icon="🛣️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
.stApp { background-color: #0f1117; color: #e0e0e0; }

/* Card style for sections */
.card {
    background: #1e2130;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    border: 1px solid #2e3250;
}

/* Title */
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #4f8ef7, #a259ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #8892b0;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Prediction badge */
.pred-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-pothole  { background: #ff4b4b33; color: #ff4b4b; border: 1.5px solid #ff4b4b; }
.badge-crack    { background: #ffa50033; color: #ffa500; border: 1.5px solid #ffa500; }
.badge-manhole  { background: #4f8ef733; color: #4f8ef7; border: 1.5px solid #4f8ef7; }
.badge-default  { background: #a259ff33; color: #a259ff; border: 1.5px solid #a259ff; }

/* Confidence bar background */
.conf-label { font-size: 0.85rem; color: #8892b0; margin-bottom: 0.2rem; }

/* Divider */
hr { border-color: #2e3250; }

/* Upload zone */
[data-testid="stFileUploadDropzone"] {
    background: #1e2130 !important;
    border: 2px dashed #4f8ef7 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE     = 128
# Paths resolved relative to this script (works locally and on Streamlit Cloud)
_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(_DIR, "road_damage_cnn.keras")
ENCODER_PATH = os.path.join(_DIR, "label_encoder.pkl")
# Write extracted weights to /tmp (always writable, even on Streamlit Cloud)
WEIGHTS_TMP  = os.path.join("/tmp", "road_damage_model.weights.h5")

# Badge colour map
BADGE_CLASS = {
    "pothole": "badge-pothole",
    "crack":   "badge-crack",
    "manhole": "badge-manhole",
}

# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    import zipfile
    import h5py
    import numpy as np
    import keras                          # Keras 3 — no TF_USE_LEGACY_KERAS needed
    from keras import layers

    # ── Build exact architecture saved by CNN2 (3).ipynb ──────────────────────
    model = keras.Sequential([
        keras.Input(shape=(128, 128, 3), name="input_layer"),
        layers.Conv2D(32,  (3, 3), activation="relu", name="conv2d"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d"),
        layers.Conv2D(64,  (3, 3), activation="relu", name="conv2d_1"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
        layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_2"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(3,   activation="softmax", name="dense_1"),
    ])

    # ── Extract model.weights.h5 from the .keras zip to /tmp ─────────────────
    with zipfile.ZipFile(MODEL_PATH, "r") as z:
        with z.open("model.weights.h5") as src, open(WEIGHTS_TMP, "wb") as dst:
            dst.write(src.read())

    # ── Load weights by layer name (Keras 3 HDF5 layout) ─────────────────────
    with h5py.File(WEIGHTS_TMP, "r") as f:
        layers_group = f["layers"]
        for layer in model.layers:
            if layer.name not in layers_group:
                continue
            vars_group = layers_group[layer.name].get("vars", {})
            for i, weight in enumerate(layer.weights):
                key = str(i)
                if key in vars_group:
                    weight.assign(np.array(vars_group[key]))

    # ── Load label encoder ────────────────────────────────────────────────────
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    return model, le


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype("float32")


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛣️ Road Damage Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a road image · Get instant AI-powered damage classification</div>', unsafe_allow_html=True)

# ── Check model files ──────────────────────────────────────────────────────────
model_ready = os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)

if not model_ready:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.warning("⚠️ **Model files not found.** Place these files in the same folder as `app.py`:")
    st.code(
        "road_damage_cnn.keras   ← trained model\n"
        "label_encoder.pkl       ← label encoder",
        language="text"
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    model, label_encoder = load_model_and_encoder()
    CLASS_NAMES = list(label_encoder.classes_)

# ── Upload section ─────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📁 Upload Road Image")
uploaded = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Main prediction flow ───────────────────────────────────────────────────────
if uploaded is not None:
    pil_img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    # ── Left: Image Preview ───────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Image Preview")
        st.image(pil_img, width=None, caption=uploaded.name)
        w, h = pil_img.size
        st.caption(f"Resolution: {w} × {h} px  |  Format: {pil_img.format or 'N/A'}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right: Prediction ─────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Prediction")

        if not model_ready:
            st.error("Model not loaded. See instructions above.")
        else:
            with st.spinner("Analysing image…"):
                tensor = preprocess(pil_img)
                probs  = model.predict(tensor, verbose=0)[0]

            pred_idx   = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx]) * 100

            # Badge
            badge_cls = BADGE_CLASS.get(pred_label.lower(), "badge-default")
            st.markdown(
                f'<span class="pred-badge {badge_cls}">{pred_label.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")
            st.progress(confidence / 100)

            st.markdown("---")
            st.markdown("**All class scores:**")
            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
                pct = float(prob) * 100
                icon = "🔹" if i != pred_idx else "✅"
                st.markdown(f'<span class="conf-label">{icon} {cls.capitalize()}</span>', unsafe_allow_html=True)
                st.progress(pct / 100, text=f"{pct:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Severity note ──────────────────────────────────────────────────────────
    if model_ready:
        severity = {
            "pothole": ("🔴 High Risk",    "Potholes pose immediate danger to vehicles and pedestrians. Urgent repair recommended."),
            "crack":   ("🟡 Moderate Risk","Surface cracks indicate early road degradation. Schedule maintenance soon."),
            "manhole": ("🔵 Low Risk",     "Manhole detected. Verify cover integrity and surrounding road condition."),
        }
        lvl, msg = severity.get(pred_label.lower(), ("⚪ Unknown", "Unable to determine severity."))
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### {lvl} — Severity Assessment")
        st.info(msg)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(
        '<div style="text-align:center; color:#8892b0; padding: 2rem 0;">⬆️ Upload an image above to begin analysis</div>',
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#555; font-size:0.8rem;">'
    'Road Damage Detection · CNN Model · Built with Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
