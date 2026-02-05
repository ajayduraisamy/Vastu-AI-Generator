import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from peft import LoraConfig, get_peft_model

# ===============================
# PATHS (AUTO)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "vastu-lora")

MODEL_ID = "stabilityai/sd-turbo"  
EPOCHS = 2
BATCH_SIZE = 1
LR = 1e-4

device = "cpu"  

# ===============================
# DATASET
# ===============================
class VastuDataset(Dataset):
    def __init__(self, metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_ID, subfolder="tokenizer"
        )

    def __len__(self):
        return min(300, len(self.data)) 

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(os.path.join(IMAGE_DIR, item["image"])).convert("RGB")
        image = image.resize((512, 512))

        image = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
             .view(512, 512, 3)
             .numpy())
        ).float() / 255.0

        image = image.permute(2, 0, 1)

        tokens = self.tokenizer(
            item["prompt"],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return image, tokens.input_ids[0]

# ===============================
# LOAD PIPELINE
# ===============================
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
).to(device)

pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

# ===============================
# APPLY LORA
# ===============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.1,
    bias="none"
)

pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.train()

# ===============================
# TRAIN LOOP 
# ===============================
dataset = VastuDataset(METADATA_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LR)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for step, (images, input_ids) in enumerate(loader):
        images = images.to(device)
        input_ids = input_ids.to(device)

       
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],)).long()

        noisy_latents = latents + noise
        encoder_hidden_states = pipe.text_encoder(input_ids)[0]

        noise_pred = pipe.unet(
            noisy_latents, timesteps, encoder_hidden_states
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 20 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

# ===============================
# SAVE LORA
# ===============================
os.makedirs(OUTPUT_DIR, exist_ok=True)
pipe.unet.save_pretrained(OUTPUT_DIR)

print("\n Training completed successfully")
print(f" LoRA saved at: {OUTPUT_DIR}")
