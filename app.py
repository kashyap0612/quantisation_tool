import streamlit as st

from scripts.utils.hf_search import search_models
from scripts.pipeline import quantize_model


st.set_page_config(
    page_title="HF Model Quantizer",
    layout="wide"
)

st.title("Hugging Face Model Quantizer")
st.caption(
    "Search → Select → Quantize vision models "
    "to TFLite (FP16 / Dynamic INT8)"
)

query = st.text_input(
    "Search Hugging Face Model",
    placeholder="mobilenet"
)

results = []

if query:

    with st.spinner("Searching..."):
        results = search_models(query)

    model_options = [
        model["id"]
        for model in results
    ]

    if model_options:

        selected_model = st.selectbox(
            "Select Model",
            model_options
        )

        quantization = st.radio(
            "Quantization Type",
            ["fp16", "int8"],
            horizontal=True
        )

        if st.button("Generate Quantized Model"):

            try:
                with st.spinner(
                    "Running pipeline..."
                ):

                    result = quantize_model(
                        selected_model,
                        quantization
                    )

                st.success(
                    "✓ Quantization Complete"
                )

                # Display benchmark results
                benchmark = result["benchmark"]
                similarity = benchmark[
                    "cosine_similarity"
                ]
                quality = benchmark["quality"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Cosine Similarity",
                        value=f"{similarity:.4f}"
                    )

                with col2:
                    if quality == "Excellent":
                        st.metric(
                            label="Quality",
                            value=f"🟢 {quality}"
                        )
                    elif quality == "Good":
                        st.metric(
                            label="Quality",
                            value=f"🟡 {quality}"
                        )
                    elif quality == "Acceptable":
                        st.metric(
                            label="Quality",
                            value=f"🟠 {quality}"
                        )
                    else:
                        st.metric(
                            label="Quality",
                            value=f"🔴 {quality}"
                        )

                st.code(
                    result["tflite_path"]
                )

                with open(
                    result["tflite_path"],
                    "rb"
                ) as f:

                    st.download_button(
                        label="Download TFLite Model",
                        data=f,
                        file_name=result[
                            "tflite_path"
                        ].split("\\")[-1],
                        mime="application/octet-stream"
                    )

            except ValueError as e:
                st.error(
                    f"⛔ Architecture Not Supported\n\n"
                    f"{e}"
                )

            except RuntimeError as e:
                st.error(
                    f"⚠️ Conversion Failed\n\n{e}"
                )

            except Exception as e:
                st.error(
                    f"❌ Unexpected Error\n\n{e}"
                )

    else:
        st.warning("No models found.")