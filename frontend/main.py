# ui.py
import streamlit as st
import requests
from PIL import Image
import io

# ==== CONFIG ====
API_BASE = "http://localhost:8000"  # change to your FastAPI base URL

st.set_page_config(page_title="PhishGuard AI", layout="centered")

# ==== CUSTOM CSS ====
st.markdown("""
<style>
    body {
        background-color: #f8f9fa;
    }
    .stTabs [role="tablist"] button {
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }
    .result-label {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .confidence {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ==== HEADER ====
st.markdown("<h1 style='text-align:center;'>🎯 PhishGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>Detect phishing in text & images, and generate content with AI</p>", unsafe_allow_html=True)

# ==== TABS ====
tab1, tab2, tab3, tab4 = st.tabs(["📝 Classify Text", "🖼 Classify Image", "🔀 Classify Both", "🎨 Generate"])

# ==== TEXT CLASSIFICATION ====
with tab1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        text_input = st.text_area("Enter text to classify:", height=150)
        if st.button("Classify Text"):
            if text_input.strip():
                resp = requests.post(f"{API_BASE}/classify-text", data={"text": text_input})
                if resp.ok:
                    result = resp.json()
                    st.markdown(f"<p class='result-label'>{result['label'].title()}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='confidence'>Confidence: {result['confidence']:.2%}</p>", unsafe_allow_html=True)
                else:
                    st.error("Error calling API")
            else:
                st.warning("Please enter some text.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==== IMAGE CLASSIFICATION ====
with tab2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        img_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
        if st.button("Classify Image"):
            if img_file:
                resp = requests.post(f"{API_BASE}/classify-image", files={"file": img_file})
                if resp.ok:
                    result = resp.json()
                    st.markdown(f"<p class='result-label'>{result['label'].title()}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='confidence'>Confidence: {result['confidence']:.2%}</p>", unsafe_allow_html=True)
                else:
                    st.error("Error calling API")
            else:
                st.warning("Please upload an image.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==== BOTH ====
with tab3:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        text_input_both = st.text_area("Enter text:", height=120, key="both_text")
        img_file_both = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"], key="both_img")
        if st.button("Classify Text & Image"):
            if text_input_both.strip() and img_file_both:
                resp = requests.post(
                    f"{API_BASE}/classify-text-image",
                    data={"text": text_input_both},
                    files={"file": img_file_both}
                )
                if resp.ok:
                    result = resp.json()
                    st.subheader(f"Overall: {result['label'].title()} ({result['confidence']:.2%})")
                    st.markdown(f"**Text:** {result['text']['label'].title()} ({result['text']['confidence']:.2%})")
                    st.markdown(f"**Image:** {result['image']['label'].title()} ({result['image']['confidence']:.2%})")
                else:
                    st.error("Error calling API")
            else:
                st.warning("Please provide both text and an image.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==== GENERATION ====
with tab4:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        gen_type = st.radio("What do you want to generate?", ["Text", "Image"], horizontal=True)
        if gen_type == "Text":
            prompt = st.text_area("Enter prompt for text generation:")
            if st.button("Generate Text"):
                if prompt.strip():
                    resp = requests.post(f"{API_BASE}/generate-text", data={"prompt": prompt})
                    if resp.ok:
                        result = resp.json()
                        st.text_area("Generated Text", result["generated_text"], height=200)
                    else:
                        st.error("Error calling API")
        else:
            prompt = st.text_input("Enter prompt for image generation:")
            if st.button("Generate Image"):
                if prompt.strip():
                    resp = requests.post(f"{API_BASE}/generate-image", data={"prompt": prompt})
                    if resp.ok:
                        result = resp.json()
                        img_url = result["image_url"]
                        try:
                            image_data = requests.get(img_url).content
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption="Generated Image", use_column_width=True)
                        except:
                            st.error("Could not load generated image.")
                    else:
                        st.error("Error calling API")
        st.markdown("</div>", unsafe_allow_html=True)
