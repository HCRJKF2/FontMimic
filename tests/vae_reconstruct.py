import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL


VAE_PATH = "./models/sdxl-vae"
INPUT_DIR = "./outputs/vae_input"
OUTPUT_DIR = "./outputs/vae_reconstruction"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    vae = AutoencoderKL.from_pretrained(VAE_PATH)
    vae.requires_grad_(False)
    vae = vae.to(DEVICE)
    vae.eval()

    tf = transforms.Compose([
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    for fname in sorted(os.listdir(INPUT_DIR)):
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            print(f"skip non-image: {fname}")
            continue

        tensor = tf(img).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

        with torch.no_grad():
            latents = vae.encode(tensor).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            recon = vae.decode(latents / vae.config.scaling_factor).sample

        recon = (recon / 2 + 0.5).clamp(0, 1)
        tensor = (tensor / 2 + 0.5).clamp(0, 1)

        # [1, 3, H, W] -> [H, W, 3] -> numpy
        orig_np = tensor[0].permute(1, 2, 0).cpu().numpy()
        recon_np = recon[0].permute(1, 2, 0).cpu().numpy()

        concat = np.concatenate([orig_np, recon_np], axis=1)  # horizontal
        concat_img = Image.fromarray((concat * 255).astype(np.uint8))

        concat_img.save(os.path.join(OUTPUT_DIR, fname))
        print(f"saved: {fname}")


if __name__ == "__main__":
    main()
