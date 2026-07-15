import os
import streamlit as st

from scripts.utils.hf_search import search_models
from scripts.pipeline import quantize_model, PIPELINE_STEPS


st.set_page_config(
    page_title="HF Model Quantizer",
    page_icon="⚡",
    layout="wide"
)

# ── Dark Theme Styling ──────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global Dark Theme ── */
    .stApp {
        background: #0a0b0f !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    html, body, [class*="st-"] {
        font-size: 17px;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Gradient Text Utility ── */
    .gradient-text {
        background: linear-gradient(135deg, #00f2fe, #4facfe, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Hero Section ── */
    .hero-section {
        text-align: center;
        padding: 60px 20px 30px 20px;
    }
    .hero-section h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        background: linear-gradient(135deg, #00f2fe, #4facfe, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-section .subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        margin: 0 0 20px 0;
        line-height: 1.7;
    }
    .hero-section .subtitle code {
        background: rgba(79, 172, 254, 0.15);
        color: #4facfe;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.95rem;
    }
    .pipeline-chip {
        display: inline-block;
        padding: 10px 24px;
        background: rgba(79, 172, 254, 0.08);
        border: 1px solid rgba(79, 172, 254, 0.25);
        border-radius: 24px;
        color: #4facfe;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* ── Search Bar ── */
    .search-section {
        max-width: 700px;
        margin: 0 auto;
        padding: 8px 0 40px 0;
    }
    .stTextInput > div > div > input {
        background: #12131a !important;
        border: 1.5px solid rgba(79, 172, 254, 0.2) !important;
        border-radius: 28px !important;
        padding: 16px 24px !important;
        font-size: 1.05rem !important;
        color: #e2e8f0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4facfe !important;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #64748b !important;
    }
    .stTextInput label {
        display: none !important;
    }

    /* ── Use-Case Cards ── */
    .usecase-grid {
        display: flex;
        gap: 16px;
        max-width: 900px;
        margin: 0 auto 40px auto;
        flex-wrap: wrap;
        justify-content: center;
    }
    .uc-card {
        flex: 1;
        min-width: 180px;
        max-width: 220px;
        background: #12131a;
        border: 1px solid rgba(79, 172, 254, 0.1);
        border-radius: 14px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .uc-card:hover {
        border-color: rgba(79, 172, 254, 0.4);
        box-shadow: 0 0 24px rgba(79, 172, 254, 0.08);
        transform: translateY(-2px);
    }
    .uc-card .uc-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .uc-card .uc-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .uc-card .uc-desc {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.5;
    }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        height: 1px;
        background: rgba(79, 172, 254, 0.1);
        margin: 8px 0 24px 0;
    }

    /* ── Selectbox, Radio, Labels ── */
    .stSelectbox label, .stRadio label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00f2fe, #4facfe) !important;
        color: #0a0b0f !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 24px rgba(79, 172, 254, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button {
        font-size: 1.05rem !important;
        border-radius: 10px !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
    }

    /* ── Status widget ── */
    [data-testid="stStatusWidget"] {
        background: #12131a !important;
        border: 1px solid rgba(79, 172, 254, 0.15) !important;
        border-radius: 12px !important;
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        color: #475569;
        font-size: 0.85rem;
        padding: 16px 0;
    }
    .footer-text span {
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ────────────────────────────────────

st.markdown("""
<div class="hero-section">
    <h1>⚡ HF Vision Model Quantizer</h1>
    <p class="subtitle">
        Compress Hugging Face vision models to lightweight
        <code>.tflite</code> format for edge deployment.<br>
        Search any model, pick your quantization, and download
        a production-ready model in minutes.
    </p>
    <div class="pipeline-chip">
        PyTorch → ONNX → TF SavedModel → Quantized TFLite
    </div>
</div>
""", unsafe_allow_html=True)

# ── Use-Case Cards ──────────────────────────────────

st.markdown("""
<div class="usecase-grid">
    <div class="uc-card">
        <div class="uc-icon">📱</div>
        <div class="uc-title">Mobile Apps</div>
        <div class="uc-desc">
            Run classifiers on Android &amp; iOS
            with TFLite runtime
        </div>
    </div>
    <div class="uc-card">
        <div class="uc-icon">🤖</div>
        <div class="uc-title">IoT &amp; Edge</div>
        <div class="uc-desc">
            Deploy on Raspberry Pi,
            Jetson Nano &amp; more
        </div>
    </div>
    <div class="uc-card">
        <div class="uc-icon">🌐</div>
        <div class="uc-title">Browser ML</div>
        <div class="uc-desc">
            Use in web apps via
            TensorFlow.js or MediaPipe
        </div>
    </div>
    <div class="uc-card">
        <div class="uc-icon">🏭</div>
        <div class="uc-title">Faster APIs</div>
        <div class="uc-desc">
            Smaller models = faster
            inference &amp; lower costs
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Centered Search Bar ─────────────────────────────

_, search_col, _ = st.columns([1, 2.5, 1])

with search_col:
    query = st.text_input(
        "Search",
        placeholder="🔍  Search Hugging Face models... "
        "e.g. mobilenet, resnet, efficientnet",
        label_visibility="collapsed"
    )

# ── Results Section ─────────────────────────────────

results = []

if query:

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    with st.spinner("🔍 Searching Hugging Face Hub..."):
        results = search_models(query)

    model_options = [
        model["id"]
        for model in results
    ]

    if model_options:

        # ── Model Selection ─────────────────────────

        col_model, col_quant = st.columns([3, 1])

        with col_model:
            selected_model = st.selectbox(
                "📋 Select Model",
                model_options
            )

        with col_quant:
            quantization = st.radio(
                "⚡ Quantization",
                ["fp16", "int8"],
                horizontal=True,
                help=(
                    "**FP16**: ~50% size reduction, "
                    "~0% accuracy loss\n\n"
                    "**INT8**: ~75% size reduction, "
                    "~1-3% accuracy loss"
                )
            )

        # ── Run Pipeline ────────────────────────────

        if st.button(
            "⚡ Generate Quantized Model",
            use_container_width=True,
            type="primary"
        ):

            try:
                status = st.status(
                    "⏳ Running quantization pipeline...",
                    expanded=True
                )

                def on_progress(step, total, icon, message):
                    """Callback: write each step to st.status."""
                    status.update(
                        label=(
                            f"⏳ Step {step}/{total}: "
                            f"{message}"
                        ),
                        state="running"
                    )
                    status.write(
                        f"{icon} **[{step}/{total}]** "
                        f"{message}"
                    )

                result = quantize_model(
                    selected_model,
                    quantization,
                    on_progress=on_progress
                )

                status.update(
                    label="✅ Pipeline Complete",
                    state="complete",
                    expanded=True
                )

                # ── Results ─────────────────────────

                st.success("✅ Quantization Complete!")

                benchmark = result["benchmark"]
                similarity = benchmark[
                    "cosine_similarity"
                ]
                quality = benchmark["quality"]

                quality_config = {
                    "Excellent": "🟢",
                    "Good": "🟡",
                    "Acceptable": "🟠",
                    "Poor": "🔴",
                }

                q_icon = quality_config.get(
                    quality, "⚪"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="📐 Cosine Similarity",
                        value=f"{similarity:.4f}"
                    )

                with col2:
                    st.metric(
                        label="🏆 Quality",
                        value=f"{q_icon} {quality}"
                    )

                with col3:
                    tflite_size = os.path.getsize(
                        result["tflite_path"]
                    )
                    size_mb = tflite_size / (1024 * 1024)
                    st.metric(
                        label="💾 Model Size",
                        value=f"{size_mb:.1f} MB"
                    )

                # ── Download ────────────────────────

                st.markdown(
                    '<hr class="custom-divider">',
                    unsafe_allow_html=True
                )

                st.code(
                    result["tflite_path"],
                    language=None
                )

                with open(
                    result["tflite_path"], "rb"
                ) as f:

                    file_name = result[
                        "tflite_path"
                    ].replace("\\", "/").split("/")[-1]

                    st.download_button(
                        label="📥 Download TFLite Model",
                        data=f,
                        file_name=file_name,
                        mime="application/octet-stream",
                        use_container_width=True,
                    )

            except ValueError as e:
                st.error(
                    f"⛔ **Architecture Not Supported**"
                    f"\n\n{e}"
                )

            except RuntimeError as e:
                st.error(
                    f"⚠️ **Conversion Failed**\n\n{e}"
                )

            except Exception as e:
                st.error(
                    f"❌ **Unexpected Error**\n\n{e}"
                )

    else:
        st.warning(
            "🔍 No models found. "
            "Try a different search term."
        )

# ── Footer ──────────────────────────────────────────

st.markdown(
    '<hr class="custom-divider">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-text">'
    '⚡ Supports <span>MobileNet · ResNet · '
    'EfficientNet · ConvNeXt · DenseNet · VGG · '
    'RegNet</span> &amp; more'
    '</div>',
    unsafe_allow_html=True
)