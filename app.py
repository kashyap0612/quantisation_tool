import os
import gradio as gr
from scripts.utils.hf_search import search_models
from scripts.pipeline import quantize_model
import spaces

# ── Custom Dark CSS (Matching Portfolio Vibe) ──
custom_css = """
body {
    background-color: #0a0b0f !important;
}
.gradio-container {
    font-family: 'Inter', sans-serif !important;
}
.header-text {
    text-align: center;
    background: linear-gradient(135deg, #00f2fe, #4facfe, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
}
.sub-text {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 5px;
}
.pipeline-chip {
    display: inline-block;
    padding: 8px 20px;
    background: rgba(79, 172, 254, 0.08);
    border: 1px solid rgba(79, 172, 254, 0.25);
    border-radius: 20px;
    color: #4facfe;
    font-size: 0.9rem;
    margin: 15px auto;
    text-align: center;
}
.center-div {
    text-align: center;
}
"""

def do_search(query):
    if not query:
        return gr.update(choices=[], value=None)
    results = search_models(query)
    options = [m["id"] for m in results]
    return gr.update(choices=options, value=options[0] if options else None)

@spaces.GPU(duration=120)
def run_quantization(model_id, quantization, progress=gr.Progress()):
    if not model_id:
        return "Please select a model first.", None, None, None, None

    # Maps pipeline steps to Gradio's live progress bar
    def on_progress(step, total, icon, message):
        progress((step, total), desc=f"{icon} {message}")

    try:
        # Run the same underlying pipeline
        result = quantize_model(
            model_id, 
            quantization, 
            on_progress=on_progress
        )
        
        benchmark = result["benchmark"]
        sim = f"{benchmark['cosine_similarity']:.4f}"
        
        q_val = benchmark["quality"]
        if q_val == "Excellent":
            quality = f"🟢 {q_val}"
        elif q_val == "Good":
            quality = f"🟡 {q_val}"
        elif q_val == "Acceptable":
            quality = f"🟠 {q_val}"
        else:
            quality = f"🔴 {q_val}"
            
        tflite_size = os.path.getsize(result["tflite_path"])
        size_mb = f"{tflite_size / (1024 * 1024):.1f} MB"
        
        success_msg = f"### ✅ Quantization Complete for `{model_id}`"
        return success_msg, sim, quality, size_mb, result["tflite_path"]
        
    except ValueError as e:
        return f"### ⛔ Architecture Not Supported\n\n{str(e)}", None, None, None, None
    except RuntimeError as e:
        return f"### ⚠️ Conversion Failed\n\n{str(e)}", None, None, None, None
    except Exception as e:
        return f"### ❌ Unexpected Error\n\n{str(e)}", None, None, None, None

# Dark theme configuration
theme = gr.themes.Monochrome(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#0a0b0f",
    body_background_fill_dark="#0a0b0f",
    block_background_fill="#12131a",
    block_background_fill_dark="#12131a",
    block_border_color="rgba(79, 172, 254, 0.2)",
    block_border_width="1px",
    button_primary_background_fill="linear-gradient(135deg, #00f2fe, #4facfe)",
    button_primary_background_fill_dark="linear-gradient(135deg, #00f2fe, #4facfe)",
    button_primary_text_color="#0a0b0f"
)

with gr.Blocks(title="HF Model Quantizer", theme=theme, css=custom_css) as demo:
    
    gr.HTML("""
    <div class="center-div">
        <h1 class="header-text">⚡ HF Vision Model Quantizer</h1>
        <div class="sub-text">Compress Hugging Face vision models to lightweight `.tflite` format for edge deployment.</div>
        <div class="pipeline-chip">PyTorch → ONNX → TF SavedModel → Quantized TFLite</div>
    </div>
    """)
    
    with gr.Row():
        query_input = gr.Textbox(
            label="🔍 Search Hugging Face Models", 
            placeholder="e.g. mobilenet, resnet, efficientnet..."
        )
        search_btn = gr.Button("Search")
        
    with gr.Row():
        model_dropdown = gr.Dropdown(label="📋 Select Model", choices=[])
        quant_radio = gr.Radio(
            label="⚡ Quantization", 
            choices=["fp16", "int8"], 
            value="fp16",
            info="FP16: ~50% smaller. INT8: ~75% smaller."
        )
        
    search_btn.click(fn=do_search, inputs=query_input, outputs=model_dropdown)
    query_input.submit(fn=do_search, inputs=query_input, outputs=model_dropdown)
    
    run_btn = gr.Button("🚀 Generate Quantized Model", variant="primary")
    
    with gr.Group():
        status_text = gr.Markdown("")
        with gr.Row():
            sim_out = gr.Textbox(label="📐 Cosine Similarity")
            quality_out = gr.Textbox(label="🏆 Quality")
            size_out = gr.Textbox(label="💾 Model Size")
        download_out = gr.File(label="📥 Download TFLite Model")
        
    run_btn.click(
        fn=run_quantization,
        inputs=[model_dropdown, quant_radio],
        outputs=[status_text, sim_out, quality_out, size_out, download_out]
    )

if __name__ == "__main__":
    demo.launch()