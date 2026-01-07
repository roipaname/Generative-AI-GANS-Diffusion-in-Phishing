import os
import torch
from diffusers import StableDiffusionPipeline
import tiktoken
from config.constant import GPT_CONFIG
from src.models.text.gpt_classifier import GPTModel,GPTForClassification

# -------------------------
# Settings
# -------------------------
OUTPUT_DIR = "generated_phishing_images"
NUM_IMAGES = 500
MODEL_ID = "runwayml/stable-diffusion-v1-5"
FINETUNED_PATH = "./outputs/inetuned-phishing-sd"
GPT_MODEL_PATH = "./outputs/gpt_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load GPT language model
# -------------------------
print("Loading custom GPT model...")
gpt_language_model = GPTModel(GPT_CONFIG).to(DEVICE)
gpt_language_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=DEVICE))
gpt_language_model.eval()

# Use GPT-2 tokenizer via tiktoken
enc = tiktoken.get_encoding("gpt2")

def generate_caption(seed_text="phishing email image", max_length=30):
    """Generate phishing-related captions using custom GPT + tiktoken"""
    # Encode seed text
    input_ids = torch.tensor([enc.encode(seed_text)], dtype=torch.long).to(DEVICE)

    generated = input_ids.clone()
    gpt_language_model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            logits = gpt_language_model(generated)[:, -1, :]  # last token
            probs = torch.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            # Stop if we hit EOS (50256 is GPT2's end-of-text token)
            if next_token.item() == 50256:
                break

    # Decode back to text
    return enc.decode(generated[0].tolist())

# -------------------------
# Load Stable Diffusion
# -------------------------
print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    unet=StableDiffusionPipeline.from_pretrained(FINETUNED_PATH, subfolder="unet").unet,
    torch_dtype=torch.float32
).to(DEVICE)

# -------------------------
# Generate phishing images
# -------------------------
print("Starting generation loop...")
for i in range(NUM_IMAGES):
    caption = generate_caption("phishing email image")
    print(f"[{i+1}/{NUM_IMAGES}] Prompt: {caption}")

    image = pipe(
        caption,
        guidance_scale=9,
        num_inference_steps=50,
        width=512,
        height=512
    ).images[0]

    save_path = os.path.join(OUTPUT_DIR, f"phishing_{i+1:03d}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")

print("All phishing images generated.")
