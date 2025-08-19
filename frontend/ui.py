# ui.py
import streamlit as st
import requests
from PIL import Image
import iogh
# ==== CONFIG ====
API_BASE = "http://localhost:8000"  # change to your FastAPI base URL

st.set_page_config(
    page_title="PhishGuard AI", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==== MODERN CSS STYLING ====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .block-container {
        max-width: 900px;
        padding: 1rem 2rem 3rem;
        background: transparent;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 2px 20px rgba(0,0,0,0.1);
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Tab styling */
    .stTabs {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: rgba(255,255,255,0.7);
        font-weight: 500;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.1);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Card styling */
    .modern-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 25px 80px rgba(0,0,0,0.15);
    }
    
    /* Input styling */
    .stTextArea textarea,
    .stTextInput input {
        border: 2px solid rgba(102,126,234,0.2);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        font-size: 1rem;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
            color:black;
    }
    
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        background: white;
        color:black;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(102,126,234,0.3);
        min-height: 3rem;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102,126,234,0.4);
        filter: brightness(1.05);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(102,126,234,0.3);
        border-radius: 16px;
        padding: 2rem;
        background: rgba(102,126,234,0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(102,126,234,0.5);
        background: rgba(102,126,234,0.08);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(102,126,234,0.05);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Results styling */
    .result-container {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102,126,234,0.2);
    }
    
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .confidence {
        font-size: 1rem;
        color: #718096;
        font-weight: 500;
    }
    
    .phishing {
        background: linear-gradient(135deg, rgba(245,101,101,0.1) 0%, rgba(237,137,54,0.1) 100%);
        border-color: rgba(245,101,101,0.3);
    }
    
    .legitimate {
        background: linear-gradient(135deg, rgba(72,187,120,0.1) 0%, rgba(56,178,172,0.1) 100%);
        border-color: rgba(72,187,120,0.3);
    }
    
    /* Success/Error messages */
    .stSuccess,
    .stError,
    .stWarning {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #2d3748;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Image display */
    .stImage {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .modern-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .block-container {
            padding: 1rem;
        }
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
tab1, tab2, tab3, tab4,tab5 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "🔍 Combined Analysis", "✨ AI Image Generator","✨ AI Code Generator"])

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
                                <div style='font-size: 1.1rem; font-weight: 600;'>{text_icon} Text Analysis</div>
                                <div>{result['text']['label'].title()}</div>
                                <div class='confidence'>{result['text']['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            img_label = result['image']['label'].lower()
                            img_class = "phishing" if img_label == "phishing" else "legitimate"
                            img_icon = "⚠️" if img_label == "phishing" else "✅"
                            
                            st.markdown(f"""
                            <div class='result-container {img_class}' style='margin-left: 0.5rem;'>
                                <div style='font-size: 1.1rem; font-weight: 600;'>{img_icon} Image Analysis</div>
                                <div>{result['image']['label'].title()}</div>
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

# ==== AI GENERATION ====
with tab4:
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("### ✨ AI Content Generator")
    st.markdown("Generate text and images using advanced AI models for testing and demonstration purposes.")
    
    gen_type = st.radio(
        "Choose generation type:",
        ["📝 Text Generation", "🎨 Image Generation"],
        horizontal=True
    )
    
    if gen_type == "📝 Text Generation":
        prompt = st.text_area(
            "Enter your text generation prompt:",
            height=120,
            placeholder="Describe what kind of text you want to generate..."
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_text_btn = st.button("✨ Generate Text", key="gen_text_btn")
        
        if generate_text_btn:
            if prompt.strip():
                with st.spinner("Generating text..."):
                    try:
                        resp = requests.post(f"{API_BASE}/generate-text", data={"prompt": prompt})
                        if resp.ok:
                            result = resp.json()
                            st.markdown("#### 📄 Generated Content")
                            st.text_area(
                                "Generated Text:",
                                result["generated_text"],
                                height=250,
                                key="generated_text_output"
                            )
                        else:
                            st.error("🚨 Error generating text. Please try again.")
                    except Exception as e:
                        st.error("🚨 Network error. Please try again.")
            else:
                st.warning("📝 Please enter a prompt for text generation.")
    
    else:  # Image Generation
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
                            except Exception as img_error:
                                st.error("🚨 Could not load the generated image.")
                        else:
                            st.error("🚨 Error generating image. Please try again.")
                    except Exception as e:
                        st.error("🚨 Network error. Please try again.")
            else:
                st.warning("🎨 Please enter a prompt for image generation.")
    
    st.markdown("</div>", unsafe_allow_html=True)
with tab5:
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("### ✨ AI Code Generator")
    st.markdown("Generate Python code snippets using an advanced AI model.")

    prompt = st.text_area(
        "Enter your Python code generation prompt:",
        height=120,
        placeholder="Describe the code you want to generate..."
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_code_btn = st.button("✨ Generate Code", key="gen_code_btn")

    if generate_code_btn:
        if prompt.strip():
            with st.spinner("Generating Python code..."):
                try:
                    resp = requests.post(f"{API_BASE}/generate-python-code", data={"prompt": prompt})
                    if resp.ok:
                        result = resp.json()
                        st.markdown("#### 📄 Generated Python Code")
                        st.text_area(
                            "Generated Code:",
                            result.get("generated_code", ""),
                            height=250,
                            key="generated_code_output"
                        )
                    else:
                        st.error("🚨 Error generating Python code. Please try again.")
                except Exception as e:
                    st.error("🚨 Network error. Please try again.")
        else:
            st.warning("📝 Please enter a prompt for Python code generation.")

    st.markdown("</div>", unsafe_allow_html=True)



# ==== FOOTER ====
st.markdown("""
<div style='text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.6);'>
    <p>🛡️ PhishGuard AI - Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)