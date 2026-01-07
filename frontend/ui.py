# ui.py
import streamlit as st
import requests
from PIL import Image
import io
import os
import re
import base64
# ==== LOCAL MODULE IMPORTS ====
from speech_to_text import transcribe_audio, stop_recording, start_recording

from url_scanner import scan_url, get_analysis_result, interpret_results
from pdf_scanner import PDFScanner
from email_alerts import EmailAlerter
from data_logger import save_analysis_data

# ==== CONFIG ====
API_BASE = "http://localhost:8000"  
logo_path = "./frontend/images/PhishGuard_logo.png"  
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon=logo_path,  
    layout="centered",
    initial_sidebar_state="collapsed",
)




# ==== LOAD DARK MODE CSS ====
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("./frontend/style.css")

# ==== HELPER FUNCTIONS ====
import re
from typing import List

def extract_urls(text: str, add_scheme: bool = False) -> List[str]:
    """
    Extract URLs from text. Matches:
      - http://... and https://...
      - www.example.com/...
      - bare domains like example.com or sub.example.co.za/path?query=1
    Args:
      text: input string
      add_scheme: if True, prepend 'http://' to domain-only matches (no scheme)
    Returns:
      list of unique URLs (order preserved)
    """
    # Regex: either scheme://... or www.... or bare domains with common TLD pattern
    url_pattern = re.compile(
        r"""
        \b(                                  # word boundary + start capture
          (?:https?://[^\s<>\"'()]+)         # 1) http(s)://...
          |(?:www\.[^\s<>\"'()]+)            # 2) www....
          |(?:[A-Za-z0-9-]+\.(?:[A-Za-z]{2,63})(?:\.[A-Za-z]{2,63})?(?:/[^\s<>\"'()]*)?)  # 3) bare domain (e.g. whatsapp.com/path)
        )\b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    raw_matches = [m.group(0) for m in url_pattern.finditer(text)]

    cleaned = []
    seen = set()
    for u in raw_matches:
        # strip trailing punctuation characters commonly attached to URLs in text
        u = u.rstrip('.,;:)\]}\'"<>')

        
        if add_scheme and not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', u):
            if not u.lower().startswith('www.'):
                u = 'http://' + u
            else:
                u = 'http://' + u  # www.* -> http://www.*

        # keep unique in order
        if u not in seen:
            seen.add(u)
            cleaned.append(u)

    return cleaned


def analyze_text(text):
    """Analyze text for phishing"""
    try:
        resp = requests.post(f"{API_BASE}/classify-text", data={"text": text})
        if resp.ok:
            return resp.json()
    except Exception as e:
        st.error(f"🚨 Error analyzing text: {str(e)}")
    return None

def analyze_image(img_file):
    """Analyze image for phishing"""
    try:
        resp = requests.post(f"{API_BASE}/classify-image", files={"file": img_file})
        if resp.ok:
            return resp.json()
    except Exception as e:
        st.error(f"🚨 Error analyzing image: {str(e)}")
    return None

def analyze_combined(text, img_file):
    """Analyze both text and image"""
    try:
        resp = requests.post(
            f"{API_BASE}/classify-text-image",
            data={"text": text},
            files={"file": img_file}
        )
        if resp.ok:
            return resp.json()
    except Exception as e:
        st.error(f"🚨 Error analyzing combined content: {str(e)}")
    return None

def scan_url_analysis(url):
    """Scan URL using VirusTotal"""
    analysis_id = scan_url(url)
    if analysis_id:
        result_data = get_analysis_result(analysis_id)
        if result_data:
            return interpret_results(result_data)
    return None

def display_result(result, result_type="Overall"):
    """Display analysis result with consistent styling"""
    if result:
        label = result.get('label', 'unknown').lower()
        confidence = result.get('confidence', 0)
        print(label)
        print(confidence)
        result_class = "phishing" if "not phishing" not  in label else "legitimate"
        icon = "✅" if "not phishing" in label else "⚠️"
        
        st.markdown(f"""
        <div class='result-container {result_class}'>
            <div class='result-label'>{icon} {result_type}: {label.title()}</div>
            <div class='confidence'>Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        return label, confidence
    return None, None

def display_url_result(result):
    """Display URL scan result"""
    if result:
        verdict = result["verdict"]
        icon = "⚠️" if "phishing" in verdict.lower() or "malicious" in verdict.lower() else "✅"
        result_class = "phishing" if "phishing" in verdict.lower() or "malicious" in verdict.lower() else "legitimate"

        st.markdown(f"""
        <div class='result-container {result_class}'>
            <div class='result-label'>{icon} {verdict}</div>
            <div class='confidence'>
                Malicious: {result['malicious']} | 
                Suspicious: {result['suspicious']} | 
                Harmless: {result['harmless']} | 
                Undetected: {result['undetected']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==== HEADER ====
# ==== HEADER WITH LOGO AND TITLE ====

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div class='main-header' style='display: flex; align-items: center; gap: 18px; margin-bottom: 1.5rem;'>
        <img src='data:image/png;base64,{logo_base64}' width='150' style='border-radius: 10px;'>
        <div>
            <h1 class='main-title' style='margin: 0;'>PhishGuard AI</h1>
            <p class='main-subtitle' style='margin: 0; font-size: 0.95rem; color: rgba(255,255,255,0.75);'>
                Advanced AI-powered phishing detection and content generation
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='main-header'>
        <h1 class='main-title'>🛡️ PhishGuard AI</h1>
        <p class='main-subtitle'>Advanced AI-powered phishing detection and content generation</p>
    </div>
    """, unsafe_allow_html=True)


# ==== MAIN UNIFIED INTERFACE ====
st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
st.markdown("### 🔍 Comprehensive Phishing Analysis")
st.markdown("Analyze text, images, URLs, or voice recordings for phishing threats. Submit any combination of inputs for complete protection.")

# Initialize session state
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "manual"  # manual or voice

# ==== INPUT MODE SELECTION ====
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("📝 Manual Input Mode", use_container_width=True, 
                 type="primary" if st.session_state.analysis_mode == "manual" else "secondary"):
        st.session_state.analysis_mode = "manual"
with col2:
    if st.button("🎙️ Voice Recording Mode", use_container_width=True,
                 type="primary" if st.session_state.analysis_mode == "voice" else "secondary"):
        st.session_state.analysis_mode = "voice"

st.markdown("---")



# ==== MANUAL INPUT MODE ====
if st.session_state.analysis_mode == "manual":
    st.markdown("#### 📝 Text Input (Optional)")
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Paste suspicious text, emails, or messages here...",
        key="main_text_input"
    )

    st.markdown("#### 🖼️ Upload File(Image or Pdf) (Optional)")
    uploaded_file = st.file_uploader(
        "Upload an image or pdf:",
        type=["png", "jpg", "jpeg","pdf"],
        help="Supported formats: PNG, JPG, JPEG, PDF",
        key="main_img_upload"
    )
    pdf_result = None
    images_from_pdf = []

    if uploaded_file and uploaded_file.type != "application/pdf":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(uploaded_file, caption="Uploaded file", width=250)

    # Analyze button
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analyze Content", key="main_analyze_btn", use_container_width=True)

    if analyze_btn:
        has_text = text_input and text_input.strip()
        has_file = uploaded_file is not None
        is_pdf = has_file and uploaded_file.type == "application/pdf"

        if not has_text and not has_file:
            st.warning("📝 Please provide text, image, or PDF.")
        else:
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            results = {}
            
            with st.spinner("🔍 Analyzing content..."):
                if is_pdf:
                    st.info("📄 Scanning PDF contents...")
                    pdf_scanner = PDFScanner()
                    pdf_result = pdf_scanner.scan_pdf(uploaded_file)

                    # Analyze extracted text
                    if pdf_result['text']:
                        st.markdown("#### 🧠 PDF Text Analysis")
                        text_result = analyze_text(pdf_result['text'])
                        if text_result:
                            label, conf = display_result(text_result, "PDF Text")
                            results['pdf_text'] = (label, conf)

                    # Analyze extracted images
                    if pdf_result['images']:
                        st.markdown("#### 🖼️ Extracted PDF Images Analysis")
                        for i, img in enumerate(pdf_result['images'], 1):
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            buf.seek(0)
                            st.image(buf, caption=f"Extracted Image {i}", width=250)
                            img_result = analyze_image(buf)
                            if img_result:
                                display_result(img_result, f"PDF Image {i}")
                    
                    # URL scan from PDF text
                    if pdf_result['urls']:
                        st.markdown("#### 🔗 URLs Found in PDF")
                        for idx, url in enumerate(pdf_result['urls'], 1):
                            with st.expander(f"URL {idx}: {url}", expanded=False):
                                url_result = scan_url_analysis(url)
                                if url_result:
                                    display_url_result(url_result)
                    if has_text:
                        st.markdown("#### 📝 Text Analysis")
                        text_result = analyze_text(text_input)
                        if text_result:
                           label, conf = display_result(text_result, "Text")
                           results['text'] = (label, conf)
                        
                # Analyze based on what's provided
                elif has_text and uploaded_file and not is_pdf:
                    # Combined analysis
                    st.markdown("#### 🔄 Combined Analysis")
                    combined_result = analyze_combined(text_input, uploaded_file)
                    if combined_result:
                        label, conf = display_result(combined_result, "Combined Result")
                        results['combined'] = (label, conf)
                        
                        # Show individual results
                        st.markdown("#### 📝 Text Analysis")
                        text_data = combined_result.get('text', {})
                        display_result(text_data, "Text")
                        
                        st.markdown("#### 🖼️ Image Analysis")
                        image_data = combined_result.get('image', {})
                        display_result(image_data, "Image")
                
                elif has_text:
                    # Text only
                    st.markdown("#### 📝 Text Analysis")
                    text_result = analyze_text(text_input)
                    if text_result:
                        label, conf = display_result(text_result, "Text")
                        results['text'] = (label, conf)
                
                elif uploaded_file and not is_pdf:
                    # Image only
                    st.markdown("#### 🖼️ Image Analysis")
                    image_result = analyze_image(uploaded_file)
                    if image_result:
                        label, conf = display_result(image_result, "Image")
                        results['image'] = (label, conf)
                
                # URL Extraction and Scanning
                if has_text:
                    urls = extract_urls(text_input)
                    if urls:
                        st.markdown("---")
                        st.markdown("#### 🔗 Detected URLs - Real-Time Scanning")
                        st.info(f"🔍 Found {len(urls)} URL(s) in the text. Scanning with VirusTotal...")
                        
                        for idx, url in enumerate(urls, 1):
                            with st.expander(f"🔗 URL {idx}: {url[:50]}{'...' if len(url) > 50 else ''}", expanded=True):
                                with st.spinner(f"Scanning URL {idx}..."):
                                    url_result = scan_url_analysis(url)
                                    if url_result:
                                        display_url_result(url_result)
                                    else:
                                        st.warning("⚠️ Unable to scan this URL. Please try again later.")
                
                # Voice alert for results
                if results:
                    # ==== SAVE ANALYSIS DATA ====
                    

                    try:
                      save_analysis_data(
                        analysis_type=(
                        "combined" if has_text and uploaded_file 
                        else "text" if has_text 
                        else "image"
                        ),
                        text=text_input if has_text else None,
                        image_result=image_result if uploaded_file and 'image_result' in locals() else None,
                        text_result=text_result if has_text and 'text_result' in locals() else None,
                        combined_result=combined_result if has_text and uploaded_file and 'combined_result' in locals() else None,
                        urls_found=urls if has_text and 'urls' in locals() else None,
                        url_results=[url_result for url_result in locals().get('url_results', [])] if 'url_results' in locals() else None,
                        voice_data=None
                      )
                      st.success("✅ Analysis data saved successfully.")
                    except Exception as e:
                       st.warning(f"⚠️ Failed to save analysis data: {e}")

                    print(results)
                    emailSender=EmailAlerter()
                    emailSender.auto_send_if_critical(results)
                    phishing_not_detected = any("not phishing" in str(label).lower() for label, _ in results.values())
                    

# ==== VOICE RECORDING MODE ====
else:
    st.markdown("#### 🎙️ Voice Recording Analysis")
    st.markdown("Record a suspicious message or voice note, then analyze it with PhishGuard AI.")
    
    # Recording controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎤 Start Recording", use_container_width=True):
            start_recording()
            st.success("🎙️ Recording started...")
    
    with col2:
        if st.button("🛑 Stop Recording", use_container_width=True):
            temp_audio = stop_recording()
            if temp_audio:
                st.session_state["last_audio"] = temp_audio
                st.success("✅ Recording stopped!")
    
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            if "last_audio" in st.session_state:
                del st.session_state["last_audio"]
            if "transcribed_text" in st.session_state:
                del st.session_state["transcribed_text"]
            st.success("🗑️ Cleared!")

    # Display recorded audio
    if "last_audio" in st.session_state:
        st.markdown("---")
        st.markdown("#### 🔊 Recorded Audio")
        st.audio(st.session_state["last_audio"], format="audio/wav")
        
        # Transcribe button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📝 Transcribe & Analyze", use_container_width=True):
                with st.spinner("🎯 Transcribing audio..."):
                    text = transcribe_audio(st.session_state["last_audio"])
                    if text:
                        st.session_state["transcribed_text"] = text
                    else:
                        st.error("🚨 Please record once more to transcribe audio.")
    
    # Display transcription and analysis
    if "transcribed_text" in st.session_state:
        st.markdown("---")
        st.markdown("#### 📝 Transcribed Text")
        transcribed_text = st.text_area(
            "Transcription:",
            value=st.session_state["transcribed_text"],
            height=150,
            key="transcription_display"
        )
        
        # Analyze transcribed text
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        with st.spinner("🔍 Analyzing transcribed content..."):
            # Text analysis
            st.markdown("#### 📝 Text Analysis")
            text_result = analyze_text(transcribed_text)
            if text_result:
                label, conf = display_result(text_result, "Transcribed Text")
                try:
                  urls = extract_urls(transcribed_text)

                  save_analysis_data(
                    analysis_type="voice",
                    text=transcribed_text,                  # the transcription text
                    image_result=None,                      # no image in this mode
                    text_result=text_result,                # AI text classification result
                    combined_result=None,                   # not applicable
                    urls_found=urls if urls else None,      # URLs from voice text
                    url_results=None,                       # URL results can be filled after scanning
                    voice_data=st.session_state.get("last_audio")  # the recorded audio file
                  )

                  st.success("✅ Voice analysis data saved successfully.")
                except Exception as e:
                 st.warning(f"⚠️ Failed to save analysis data: {e}")
                
                
                
            
            # URL scanning from transcribed text
            urls = extract_urls(transcribed_text)
            if urls:
                st.markdown("---")
                st.markdown("#### 🔗 Detected URLs in Transcription")
                st.info(f"🔍 Found {len(urls)} URL(s) in the transcribed text. Scanning...")
                
                for idx, url in enumerate(urls, 1):
                    with st.expander(f"🔗 URL {idx}: {url[:50]}{'...' if len(url) > 50 else ''}", expanded=True):
                        with st.spinner(f"Scanning URL {idx}..."):
                            url_result = scan_url_analysis(url)
                            if url_result:
                                display_url_result(url_result)
                            else:
                                st.warning("⚠️ Unable to scan this URL. Please try again later.")

st.markdown("</div>", unsafe_allow_html=True)

# ==== FOOTER ====
st.markdown("""
<div style='text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.6);'>
    <p>🛡️ PhishGuard AI — Powered by Advanced Machine Learning</p>
    <p style='font-size: 0.9em; margin-top: 0.5rem;'>Submit text, images, or voice recordings • Automatic URL detection & scanning • Real-time threat analysis</p>
</div>
""", unsafe_allow_html=True)