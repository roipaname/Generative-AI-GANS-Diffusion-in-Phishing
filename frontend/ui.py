# ui.py
import streamlit as st
import requests
from PIL import Image
import io
import os

# ==== CONFIG ====
API_BASE = "http://localhost:8000"  # change to your FastAPI base URL

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",   
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==== ROYAL GLASSMORPHIC CSS STYLING ====
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ==== HEADER ====
st.markdown("""
<div class='main-header'>
    <h1 class='main-title'>🛡️ PhishGuard AI</h1>
    <p class='main-subtitle'>Advanced AI-powered phishing detection and content generation</p>
</div>
""", unsafe_allow_html=True)

# ==== TABS ====
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "🔍 Combined Analysis", "✨ AI Image Generator", "✨ AI Code Generator"])

# ==== TEXT CLASSIFICATION ====
with tab1:
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Text Classification")
    st.markdown("Analyze text content for potential phishing indicators using advanced NLP models.")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=180,
        placeholder="Paste suspicious text, emails, or messages here..."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_text_btn = st.button("🔍 Analyze Text", key="text_btn")
    
    if classify_text_btn:
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                try:
                    resp = requests.post(f"{API_BASE}/classify-text", data={"text": text_input})
                    if resp.ok:
                        result = resp.json()
                        label = result['label'].lower()
                        confidence = result['confidence']
                        
                        result_class = "phishing" if label == "phishing" else "legitimate"
                        icon = "⚠️" if label == "phishing" else "✅"
                        
                        st.markdown(f"""
                        <div class='result-container {result_class}'>
                            <div class='result-label'>{icon} {result['label'].title()}</div>
                            <div class='confidence'>Confidence: {confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("🚨 Error connecting to API. Please check your connection.")
                except Exception as e:
                    st.error("🚨 Network error. Please try again.")
        else:
            st.warning("📝 Please enter some text to analyze.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==== IMAGE CLASSIFICATION ====
with tab2:
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Image Classification")
    st.markdown("Analyze images for phishing indicators, suspicious layouts, and fraudulent visual elements.")
    
    img_file = st.file_uploader(
        "Upload an image:",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if img_file:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(img_file, caption="Uploaded Image", width=200)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_img_btn = st.button("🔍 Analyze Image", key="img_btn")
    
    if classify_img_btn:
        if img_file:
            with st.spinner("Analyzing image..."):
                try:
                    resp = requests.post(f"{API_BASE}/classify-image", files={"file": img_file})
                    if resp.ok:
                        result = resp.json()
                        label = result['label'].lower()
                        confidence = result['confidence']
                        
                        result_class = "phishing" if label == "phishing" else "legitimate"
                        icon = "⚠️" if label == "phishing" else "✅"
                        
                        st.markdown(f"""
                        <div class='result-container {result_class}'>
                            <div class='result-label'>{icon} {result['label'].title()}</div>
                            <div class='confidence'>Confidence: {confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("🚨 Error connecting to API. Please check your connection.")
                except Exception as e:
                    st.error("🚨 Network error. Please try again.")
        else:
            st.warning("🖼️ Please upload an image to analyze.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==== COMBINED ANALYSIS ====
with tab3:
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Combined Analysis")
    st.markdown("Get comprehensive analysis by combining both text and image detection for maximum accuracy.")
    
    text_input_both = st.text_area(
        "Enter text:",
        height=120,
        key="both_text",
        placeholder="Enter associated text content..."
    )
    
    img_file_both = st.file_uploader(
        "Upload an image:",
        type=["png", "jpg", "jpeg"],
        key="both_img",
        help="Upload the image to analyze alongside the text"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_both_btn = st.button("🔍 Analyze Both", key="both_btn")
    
    if classify_both_btn:
        if text_input_both.strip() and img_file_both:
            with st.spinner("Performing comprehensive analysis..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/classify-text-image",
                        data={"text": text_input_both},
                        files={"file": img_file_both}
                    )
                    if resp.ok:
                        result = resp.json()
                        
                        # Overall result
                        overall_label = result['label'].lower()
                        overall_confidence = result['confidence']
                        overall_class = "phishing" if overall_label == "phishing" else "legitimate"
                        overall_icon = "⚠️" if overall_label == "phishing" else "✅"
                        
                        st.markdown(f"""
                        <div class='result-container {overall_class}'>
                            <div class='result-label'>{overall_icon} Overall: {result['label'].title()}</div>
                            <div class='confidence'>Combined Confidence: {overall_confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Individual results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            text_label = result['text']['label'].lower()
                            text_class = "phishing" if text_label == "phishing" else "legitimate"
                            text_icon = "⚠️" if text_label == "phishing" else "✅"
                            
                            st.markdown(f"""
                            <div class='result-container {text_class}' style='margin-right: 0.5rem;'>
                                <div style='font-size: 1.3rem; font-weight: 700;'>{text_icon} Text Analysis</div>
                                <div style='font-size: 1.1rem; margin: 0.5rem 0;'>{result['text']['label'].title()}</div>
                                <div class='confidence'>{result['text']['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            img_label = result['image']['label'].lower()
                            img_class = "phishing" if img_label == "phishing" else "legitimate"
                            img_icon = "⚠️" if img_label == "phishing" else "✅"
                            
                            st.markdown(f"""
                            <div class='result-container {img_class}' style='margin-left: 0.5rem;'>
                                <div style='font-size: 1.3rem; font-weight: 700;'>{img_icon} Image Analysis</div>
                                <div style='font-size: 1.1rem; margin: 0.5rem 0;'>{result['image']['label'].title()}</div>
                                <div class='confidence'>{result['image']['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("🚨 Error connecting to API. Please check your connection.")
                except Exception as e:
                    st.error("🚨 Network error. Please try again.")
        else:
            st.warning("📝🖼️ Please provide both text and an image for combined analysis.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==== AI IMAGE GENERATION ====
with tab4:
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("### ✨ AI Image Generator")
    st.markdown("Generate high-quality images using advanced AI models for testing and demonstration purposes.")
    
    prompt = st.text_input(
        "Enter your image generation prompt:",
        placeholder="Describe the image you want to generate..."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_img_btn = st.button("🎨 Generate Image", key="gen_img_btn")
    
    if generate_img_btn:
        if prompt.strip():
            with st.spinner("Creating your image..."):
                try:
                    resp = requests.post(f"{API_BASE}/generate-image", data={"prompt": prompt})
                    if resp.ok:
                        result = resp.json()
                        img_url = result["image_url"]
                        
                        st.markdown("#### 🖼️ Generated Image")
                        try:
                            image_data = requests.get(img_url).content
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption="Generated Image", use_column_width=True)
                        except Exception:
                            st.error("⚠️ Failed to load generated image.")
                    else:
                        st.error("🚨 Error connecting to API. Please check your connection.")
                except Exception as e:
                    st.error("🚨 Network error. Please try again.")
        else:
            st.warning("💡 Please enter a prompt to generate an image.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==== AI CODE GENERATION ====
with tab5:
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("### ✨ AI Code Generator")
    st.markdown("Generate code snippets using AI models for development and educational purposes.")
    
    code_prompt = st.text_area(
        "Enter your code generation prompt:",
        height=150,
        placeholder="Describe the code you want the AI to generate..."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_code_btn = st.button("💻 Generate Code", key="gen_code_btn")
    
    if generate_code_btn:
        if code_prompt.strip():
            with st.spinner("Generating your code..."):
                try:
                    resp = requests.post(f"{API_BASE}/generate-python-code", data={"prompt": code_prompt})
                    if resp.ok:
                        result = resp.json()
                        code_output = result.get("generated_code", "")
                        
                        if code_output:
                            st.markdown("#### 📝 Generated Code")
                            st.code(code_output, language="python")
                        else:
                            st.warning("⚠️ No code generated. Try refining your prompt.")
                    else:
                        st.error("🚨 Error connecting to API. Please check your connection.")
                except Exception as e:
                    st.error("🚨 Network error. Please try again.")
        else:
            st.warning("💡 Please enter a prompt to generate code.")
    
    st.markdown("</div>", unsafe_allow_html=True)



# ==== FOOTER ====
st.markdown("""
<div style='text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.6);'>
    <p>🛡️ PhishGuard AI - Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)