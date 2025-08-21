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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==== MODERN NEUROMORPHIC CSS STYLING ====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .block-container {
        max-width: 1000px;
        padding: 2rem 3rem 4rem;
        background: transparent;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 4rem;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        border-radius: 30px;
        box-shadow: 
            20px 20px 40px rgba(0,0,0,0.1),
            -20px -20px 40px rgba(255,255,255,0.9),
            inset 5px 5px 10px rgba(0,0,0,0.05),
            inset -5px -5px 10px rgba(255,255,255,0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.03em;
        text-shadow: none;
    }
    
    .main-subtitle {
        font-size: 1.3rem;
        color: #64748b;
        margin-top: 1rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Tab styling */
    .stTabs {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.6) 100%);
        backdrop-filter: blur(30px);
        border-radius: 25px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 
            15px 15px 30px rgba(0,0,0,0.1),
            -15px -15px 30px rgba(255,255,255,0.9),
            inset 2px 2px 5px rgba(0,0,0,0.03),
            inset -2px -2px 5px rgba(255,255,255,0.7);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.5) 100%);
        border: none;
        color: #475569;
        font-weight: 600;
        padding: 1.2rem 2rem;
        border-radius: 18px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1rem;
        box-shadow: 
            8px 8px 16px rgba(0,0,0,0.08),
            -8px -8px 16px rgba(255,255,255,0.9),
            inset 1px 1px 2px rgba(0,0,0,0.02),
            inset -1px -1px 2px rgba(255,255,255,0.6);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 
            12px 12px 24px rgba(0,0,0,0.12),
            -12px -12px 24px rgba(255,255,255,0.9),
            inset 2px 2px 4px rgba(0,0,0,0.03),
            inset -2px -2px 4px rgba(255,255,255,0.7);
        color: #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 
            10px 10px 20px rgba(102,126,234,0.2),
            -10px -10px 20px rgba(255,255,255,0.9),
            inset 2px 2px 4px rgba(255,255,255,0.2),
            inset -2px -2px 4px rgba(0,0,0,0.1) !important;
        transform: translateY(-1px);
    }
    
    /* Card styling */
    .modern-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.8) 100%);
        backdrop-filter: blur(30px);
        border-radius: 32px;
        padding: 3rem;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 
            25px 25px 50px rgba(0,0,0,0.08),
            -25px -25px 50px rgba(255,255,255,0.9),
            inset 3px 3px 6px rgba(0,0,0,0.02),
            inset -3px -3px 6px rgba(255,255,255,0.8);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 32px 32px 0 0;
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 
            30px 30px 60px rgba(0,0,0,0.12),
            -30px -30px 60px rgba(255,255,255,0.9),
            inset 4px 4px 8px rgba(0,0,0,0.03),
            inset -4px -4px 8px rgba(255,255,255,0.8);
    }
    
    /* Section headers */
    .modern-card h3 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .modern-card p {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Input styling */
    .stTextArea textarea,
    .stTextInput input {
        border: 2px solid rgba(255,255,255,0.6);
        border-radius: 20px;
        padding: 1.2rem 1.5rem;
        font-size: 1rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        color: #334155 !important;
        box-shadow: 
            inset 8px 8px 16px rgba(0,0,0,0.05),
            inset -8px -8px 16px rgba(255,255,255,0.9);
    }
    
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 
            0 0 0 3px rgba(102,126,234,0.1),
            inset 6px 6px 12px rgba(0,0,0,0.06),
            inset -6px -6px 12px rgba(255,255,255,0.9);
        background: rgba(255,255,255,0.95);
        color: #1e293b !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            12px 12px 24px rgba(102,126,234,0.2),
            -12px -12px 24px rgba(255,255,255,0.9),
            inset 2px 2px 4px rgba(255,255,255,0.2),
            inset -2px -2px 4px rgba(0,0,0,0.1);
        min-height: 3.5rem;
        width: 100%;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 
            16px 16px 32px rgba(102,126,234,0.25),
            -16px -16px 32px rgba(255,255,255,0.9),
            inset 3px 3px 6px rgba(255,255,255,0.25),
            inset -3px -3px 6px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
    }
    
    .stButton button:active {
        transform: translateY(-1px);
        box-shadow: 
            8px 8px 16px rgba(102,126,234,0.15),
            -8px -8px 16px rgba(255,255,255,0.9),
            inset 4px 4px 8px rgba(0,0,0,0.08),
            inset -4px -4px 8px rgba(255,255,255,0.3);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 3px dashed #cbd5e1;
        border-radius: 24px;
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.6) 100%);
        backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            inset 8px 8px 16px rgba(0,0,0,0.03),
            inset -8px -8px 16px rgba(255,255,255,0.9);
        text-align: center;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(255,255,255,0.8) 100%);
        box-shadow: 
            inset 6px 6px 12px rgba(102,126,234,0.05),
            inset -6px -6px 12px rgba(255,255,255,0.9);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.6) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 
            inset 6px 6px 12px rgba(0,0,0,0.03),
            inset -6px -6px 12px rgba(255,255,255,0.8);
        border: 1px solid rgba(255,255,255,0.4);
    }
    
    /* Results styling */
    .result-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid rgba(255,255,255,0.4);
        backdrop-filter: blur(20px);
        box-shadow: 
            15px 15px 30px rgba(0,0,0,0.08),
            -15px -15px 30px rgba(255,255,255,0.9),
            inset 2px 2px 4px rgba(0,0,0,0.02),
            inset -2px -2px 4px rgba(255,255,255,0.7);
        transition: all 0.4s ease;
    }
    
    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .confidence {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 600;
    }
    
    .phishing {
        background: linear-gradient(135deg, rgba(248,113,113,0.1) 0%, rgba(252,165,165,0.1) 100%);
        border-color: rgba(248,113,113,0.3);
        box-shadow: 
            15px 15px 30px rgba(248,113,113,0.1),
            -15px -15px 30px rgba(255,255,255,0.9),
            inset 2px 2px 4px rgba(248,113,113,0.05),
            inset -2px -2px 4px rgba(255,255,255,0.7);
    }
    
    .legitimate {
        background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(74,222,128,0.1) 100%);
        border-color: rgba(34,197,94,0.3);
        box-shadow: 
            15px 15px 30px rgba(34,197,94,0.1),
            -15px -15px 30px rgba(255,255,255,0.9),
            inset 2px 2px 4px rgba(34,197,94,0.05),
            inset -2px -2px 4px rgba(255,255,255,0.7);
    }
    
    /* Success/Error messages */
    .stSuccess,
    .stError,
    .stWarning {
        border-radius: 16px;
        border: none;
        backdrop-filter: blur(20px);
        box-shadow: 
            10px 10px 20px rgba(0,0,0,0.06),
            -10px -10px 20px rgba(255,255,255,0.9);
    }
    
    /* Image display */
    .stImage {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 
            15px 15px 30px rgba(0,0,0,0.1),
            -15px -15px 30px rgba(255,255,255,0.9);
    }
    
    /* Loading animation */
    @keyframes neuromorphicPulse {
        0%, 100% { 
            box-shadow: 
                8px 8px 16px rgba(0,0,0,0.1),
                -8px -8px 16px rgba(255,255,255,0.9);
        }
        50% { 
            box-shadow: 
                12px 12px 24px rgba(0,0,0,0.15),
                -12px -12px 24px rgba(255,255,255,0.9);
        }
    }
    
    .loading {
        animation: neuromorphicPulse 2s infinite;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.6) 100%);
        border-radius: 20px;
        box-shadow: 
            inset 8px 8px 16px rgba(0,0,0,0.03),
            inset -8px -8px 16px rgba(255,255,255,0.9);
        border: 1px solid rgba(255,255,255,0.4);
        color: #64748b;
        font-weight: 500;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.8rem;
        }
        
        .modern-card {
            padding: 2rem;
            margin: 1.5rem 0;
        }
        
        .block-container {
            padding: 1rem;
        }
        
        .main-header {
            padding: 2rem 1rem;
            margin-bottom: 2rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.6) 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
    }
</style>
""", unsafe_allow_html=True)

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
                    resp = requests.post(f"{API_BASE}/generate-code", data={"prompt": code_prompt})
                    if resp.ok:
                        result = resp.json()
                        code_output = result.get("code", "")
                        
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