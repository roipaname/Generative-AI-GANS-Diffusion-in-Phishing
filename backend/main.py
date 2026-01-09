import io
import os, sys, time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
#from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
#from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tiktoken

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.constant import GPT_CONFIG
from src.models.text.gpt_classifier import GPTModel, GPTForClassification
from src.utils.text_utils import text_to_token_ids, token_ids_to_text, generate_text_simple

# Import the helper
from s3_model_loader import load_cnn_model, load_torch_model, download_model_from_s3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tiktoken tokenizer for GPT
tokenizer = tiktoken.get_encoding("gpt2")

# Global models
cnn_model = None
gpt_language_model = None
gpt_classifier_model = None
code_model = None
tokenizer_code = None
pipe = None

# Path for codegen and Stable Diffusion in S3
CODEGEN_PATH = "checkpoint-8394"
STABLE_DIFFUSION_PATH = "stable-diffusion-v1-5"


# === FastAPI startup ===
@app.on_event("startup")
def load_models():
    global cnn_model, gpt_language_model, gpt_classifier_model, code_model, tokenizer_code, pipe

    # CNN Keras model
    cnn_model = load_cnn_model("cnn_model.h5")

    # GPT language model
    gpt_language_model = GPTModel(GPT_CONFIG).to(device)
    gpt_language_model.load_state_dict(
        torch.load(download_model_from_s3("gpt_model.pth"), map_location=device)
    )
    gpt_language_model.eval()
    print("GPT language model loaded ✅")

    # GPT classifier
    base_gpt = gpt_language_model
    gpt_classifier_model = GPTForClassification(
        base_gpt, hidden_size=GPT_CONFIG['emb_dim'], num_classes=2
    ).to(device)
    gpt_classifier_model.load_state_dict(
        torch.load(download_model_from_s3("classification_gpt.pth"), map_location=device)
    )
    gpt_classifier_model.eval()
    print("GPT classifier model loaded ✅")


    """

    # Code generator model
    tokenizer_code = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    code_model = AutoModelForCausalLM.from_pretrained(download_model_from_s3(CODEGEN_PATH)).to(device)
    print("Code generator model loaded ✅")

    # Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(download_model_from_s3(STABLE_DIFFUSION_PATH), torch_dtype=torch.float32)
    pipe.to("cpu")
    print("Stable Diffusion pipeline loaded ✅")"""


# === Helper functions ===

def preprocess_image(file: UploadFile) -> np.ndarray:
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    image = image.resize((128, 128))
    return np.expand_dims(np.array(image) / 255.0, axis=0)


def classify_image(file: UploadFile):
    img = preprocess_image(file)
    preds = cnn_model.predict(img)
    idx = np.argmax(preds, axis=1)[0]
    conf = float(np.max(preds))
    label = "phishing" if idx == 1 else "not phishing"
    return label, conf


def generate_text(prompt: str, max_new_tokens=50):
    gpt_language_model.eval()
    with torch.no_grad():
        encoded = text_to_token_ids(prompt, tokenizer).to(device)
        token_ids = generate_text_simple(
            gpt_language_model, idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=GPT_CONFIG['context_length']
        )
        return token_ids_to_text(token_ids, tokenizer)


def generate_code(prompt: str, max_new_tokens=10000):
    code_model.eval()
    with torch.no_grad():
        inputs = tokenizer_code(prompt, return_tensors="pt").to(device)
        output = code_model.generate(**inputs, max_length=256, max_new_tokens=max_new_tokens)
        return tokenizer_code.decode(output[0], skip_special_tokens=True)


def classify_text(text: str):
    gpt_classifier_model.eval()
    with torch.no_grad():
        encoded = text_to_token_ids(text, tokenizer).to(device)
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
        logits = gpt_classifier_model(encoded)
        probs = torch.softmax(logits, dim=-1)
        conf, pred_class = torch.max(probs, dim=-1)
    label = "phishing" if pred_class.item() == 1 else "not phishing"
    return label, conf.item()


def classify_text_and_image(text: str, file: UploadFile):
    text_label, text_conf = classify_text(text)
    image_label, image_conf = classify_image(file)
    combined_label = "phishing" if text_label == "phishing" or image_label == "phishing" else "not phishing"
    combined_confidence = max(text_conf, image_conf)
    return {
        "label": combined_label,
        "confidence": combined_confidence,
        "text": {"label": text_label, "confidence": text_conf},
        "image": {"label": image_label, "confidence": image_conf},
    }


def generate_image_diffusion(prompt: str) -> str:
    output_dir = "./outputs/generated_images"
    os.makedirs(output_dir, exist_ok=True)
    result = pipe(prompt, guidance_scale=9, num_inference_steps=70, width=512, height=512)
    image = result.images[0]
    timestamp = int(time.time())
    safe_prompt = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in prompt)[:50]
    filename = f"{safe_prompt.replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    return filepath


# === API Endpoints ===

@app.post("/generate-text")
async def api_generate_text(prompt: str = Form(...)):
    return {"generated_text": generate_text(prompt)}


@app.post("/classify-text")
async def api_classify_text(text: str = Form(...)):
    label, conf = classify_text(text)
    return {"label": label, "confidence": conf}


@app.post("/classify-image")
async def api_classify_image(file: UploadFile = File(...)):
    label, conf = classify_image(file)
    return {"label": label, "confidence": conf}


@app.post("/classify-text-image")
async def api_classify_text_image(text: str = Form(...), file: UploadFile = File(...)):
    return classify_text_and_image(text, file)

"""
@app.post("/generate-image")
async def api_generate_image(prompt: str = Form(...)):
    return {"image_url": generate_image_diffusion(prompt)}


@app.post("/generate-python-code")
async def api_generate_code(prompt: str = Form(...)):
    return {"generated_code": generate_code(prompt)}"""
