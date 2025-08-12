# Install required libraries:
# pip install diffusers transformers accelerate torch safetensors

import torch
from diffusers import StableDiffusionPipeline

# Choose a Stable Diffusion model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"

# Load the pipeline (with float16 for faster inference on GPUs)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)
pipe.save_pretrained("stable-diffusion-v1-5")
# Move model to GPU (or use CPU if you don’t have a GPU)
pipe = pipe.to("cpu")
print("Starting")
# Your text prompt
prompt = ("A flying spaceship, ultra-detailed, highly realistic, "
          "sharp focus, cinematic lighting, intricate details")

# Generate the image
image = pipe(prompt, guidance_scale=9, num_inference_steps=100,width=1080,height=1080).images[0]

# Save image to file
image.save("generated_image.png")

print("Image saved as generated_image.png")
