import io
import os,sys
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from config.constant import GPT_CONFIG


from src.models.text.gpt_classifier import GPTModel,GPTForClassification
import time

from src.utils.text_utils import text_to_token_ids, token_ids_to_text, generate_text_simple
import tiktoken
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



# Load tiktoken encoding for your GPT model (assuming GPT-2 compatible)
tokenizer = tiktoken.get_encoding("gpt2")

app = FastAPI()

# CORS middleware for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load models on startup ===
print("Loading CNN model...")
cnn_model = load_model("./outputs/cnn_model.h5")
print("CNN model loaded.")

print("Loading GPT language model...")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt_language_model = GPTModel(GPT_CONFIG).to(device)
gpt_language_model.load_state_dict(torch.load("./outputs/model_latest (1).pt", map_location=device))
gpt_language_model.eval()
print("GPT language model loaded.")

print("Loading GPT classification model...")
base_gpt=gpt_language_model
gpt_classifier_model = GPTForClassification(base_gpt, hidden_size=GPT_CONFIG['emb_dim'], num_classes=2).to(device)
gpt_classifier_model.load_state_dict(torch.load("./outputs/classification_model.pt", map_location=device))
gpt_classifier_model.eval()
print("GPT classifier model loaded.")

# === Helper functions ===

def preprocess_image(file: UploadFile) -> np.ndarray:
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    image = image.resize((128, 128))  # CNN trained on 128x128 images
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # add batch dim

def classify_image(file: UploadFile):
    img = preprocess_image(file)
    preds = cnn_model.predict(img)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    label = "phishing" if class_idx == 1 else "not phishing"
    return label, confidence

def generate_text(prompt: str, max_new_tokens=50):
    gpt_language_model.eval()
    with torch.no_grad():
        encoded = text_to_token_ids(prompt, tokenizer).to(device)  
        token_ids = generate_text_simple(
            model=gpt_language_model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=GPT_CONFIG['context_length']
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer) 
    return decoded_text

def classify_text(text: str):
    gpt_classifier_model.eval()
    with torch.no_grad():
        encoded = text_to_token_ids(text, tokenizer).to(device)
        if encoded.dim() == 1:  # ensure batch dimension
            encoded = encoded.unsqueeze(0)
        logits = gpt_classifier_model(encoded)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = torch.max(probs, dim=-1)
    label = "phishing" if pred_class.item() == 1 else "not phishing"
    return label, confidence.item()


def classify_text_and_image(text: str, file: UploadFile):
    text_label, text_conf = classify_text(text)
    image_label, image_conf = classify_image(file)
    combined_label = "phishing" if (text_label == "phishing" or image_label == "phishing") else "not phishing"
    combined_confidence = max(text_conf, image_conf)
    return {
        "label": combined_label,
        "confidence": combined_confidence,
        "text": {"label": text_label, "confidence": text_conf},
        "image": {"label": image_label, "confidence": image_conf},
    }

# Stub for diffusion image generation (work in progress)
def generate_image_diffusion(prompt: str) -> str:
    # Ensure output directory exists
    output_dir = "./outputs/generated_images"
    os.makedirs(output_dir, exist_ok=True)

    # Generate image using loaded pipeline
    with torch.no_grad():
        result = pipe(prompt, guidance_scale=9,num_inference_steps=70,width=768,height=768)
    image = result.images[0]

    # Create a unique filename with timestamp
    timestamp = int(time.time())
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)[:50]
    filename = f"{safe_prompt.replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Save image file
    image.save(filepath)

    # Return the relative path or URL to the saved image
    # You can adjust this depending on how your server serves static files
    return filepath
@app.on_event("startup")
def load_model():
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "./outputs/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    ).to("cpu")

# === API Endpoints ===

@app.post("/generate-text")
async def api_generate_text(prompt: str = Form(...)):
    generated = generate_text(prompt)
    return {"generated_text": generated}

@app.post("/classify-text")
async def api_classify_text(text: str = Form(...)):
    label, confidence = classify_text(text)
    return {"label": label, "confidence": confidence}

@app.post("/classify-image")
async def api_classify_image(file: UploadFile = File(...)):
    label, confidence = classify_image(file)
    return {"label": label, "confidence": confidence}

@app.post("/classify-text-image")
async def api_classify_text_image(text: str = Form(...), file: UploadFile = File(...)):
    result = classify_text_and_image(text, file)
    return result

@app.post("/generate-image")
async def api_generate_image(prompt: str = Form(...)):
    image_path = generate_image_diffusion(prompt)
    return {"image_url": image_path}
