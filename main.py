import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from PIL import Image
import os
import cairosvg
import io
import argparse

# =========================
# Helper: Rasterize SVG
# =========================
def rasterize_svg(svg_path: str, size: int = 256) -> Image.Image:
    """Convert an SVG file to a raster image (PIL Image)."""
    with open(svg_path, 'rb') as svg_file:
        svg_data = svg_file.read()
        png_data = cairosvg.svg2png(bytestring=svg_data, output_width=size, output_height=size)
        image = Image.open(io.BytesIO(png_data))
    return image

# =========================
# Dataset: SVG + Captions
# =========================
class SVGTextImageDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.svg_files = [f for f in os.listdir(data_dir) if f.endswith(".svg")]
        self.captions = [f.split('.')[0] for f in self.svg_files]

    def __len__(self):
        return len(self.svg_files)

    def __getitem__(self, idx):
        svg_path = os.path.join(self.data_dir, self.svg_files[idx])
        image = rasterize_svg(svg_path)  # Convert SVG to raster image
        image = image.resize((256, 256))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]

        caption = self.captions[idx]
        tokens = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {"image": image, "tokens": tokens.input_ids[0]}

# =========================
# Training Function
# =========================
def train(data_dir: str, output_dir: str, epochs: int, batch_size: int, lr: float, device: str):
    # Load tokenizer, text encoder, U-Net, and scheduler
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    unet = UNet2DConditionModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 1024),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear")

    # Prepare dataset and dataloader
    dataset = SVGTextImageDataset(data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(list(text_encoder.parameters()) + list(unet.parameters()), lr=lr)

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)

            # Text encoding
            text_embeddings = text_encoder(input_ids=tokens).last_hidden_state

            # Add noise to images
            noise = torch.randn_like(images).to(device)
            noisy_images = scheduler.add_noise(images, noise)

            # U-Net forward pass
            predicted_noise = unet(noisy_images, text_embeddings).sample
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save the model
    text_encoder.save_pretrained(os.path.join(output_dir, "text_encoder"))
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    print("Model saved!")

# =========================
# Generate Images Function
# =========================
def generate(prompt: str, output_dir: str, device: str):
    # Load tokenizer, text encoder, U-Net, and scheduler
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(output_dir, "text_encoder")).to(device)
    unet = UNet2DConditionModel.from_pretrained(os.path.join(output_dir, "unet")).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear")

    # Encode the prompt
    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
    text_embeddings = text_encoder(input_ids=tokens).last_hidden_state

    # Generate noise
    noise = torch.randn((1, 3, 256, 256)).to(device)

    # Reverse diffusion process
    for t in reversed(range(scheduler.num_train_timesteps)):
        noise = scheduler.step(unet(noise, text_embeddings).sample, t, noise).prev_sample

    # Convert to image
    image = (noise.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(image).save("generated_image.png")
    print("Image saved as 'generated_image.png'!")

# =========================
# Main Function
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Generate Images with Diffusion Models on SVG Datasets")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], required=True, help="Mode: train or generate")
    parser.add_argument("--data_dir", type=str, help="Path to SVG dataset (for training)")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to save models and results")
    parser.add_argument("--prompt", type=str, help="Prompt to generate an image (for generation)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: 'cuda' or 'cpu'")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.data_dir:
            raise ValueError("For training, --data_dir must be specified.")
        train(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr, args.device)
    elif args.mode == "generate":
        if not args.prompt:
            raise ValueError("For generating, --prompt must be specified.")
        generate(args.prompt, args.output_dir, args.device)
