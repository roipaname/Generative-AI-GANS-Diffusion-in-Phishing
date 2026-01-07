import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers import UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# 1. Model and tokenizer
model_id = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# 2. Dataset 
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = load_dataset("imagefolder", data_dir="data/phishingimage/phishing_site_1")["train"]

def preprocess(batch):
    images = [transform(img.convert("RGB")) for img in batch["image"]]
    batch["pixel_values"] = torch.stack(images)
    
    batch["input_ids"] = tokenizer(
        ["phishing email image"] * len(images),
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    return batch

dataset = dataset.with_transform(preprocess)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. Optimizer and accelerator
accelerator = Accelerator()
unet, optimizer, train_dataloader = accelerator.prepare(
    unet,
    torch.optim.AdamW(unet.parameters(), lr=1e-5),
    train_dataloader
)

noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# 4. Training loop
for epoch in range(50):  
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["pixel_values"]
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        model_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item()}")

# 5. Saving fine-tuned model
accelerator.wait_for_everyone()
unet = accelerator.unwrap_model(unet)
unet.save_pretrained("./outputs/finetuned-phishing-sd")

# 6. Generate with fine-tuned model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=UNet2DConditionModel.from_pretrained("./outputs/finetuned-phishing-sd"),
    torch_dtype=torch.float32
).to("cpu")

image = pipe("A phishing-themed illustration, cinematic, high detail").images[0]
image.save("finetuned_output.png")
print("Fine-tuned image saved.")
