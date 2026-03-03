from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from diffusion.model import GaussianDiffusion
from diffusion.model import SimpleUNet


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18


device = torch.device("cpu")
model = SimpleUNet().to(device)
model.load_state_dict(torch.load("models/ddpm_cifar10.pth", map_location=device))
diffusion = GaussianDiffusion(n_timesteps=1000, device=device)

# Generate a grid of 16 images
samples = diffusion.sample(model, shape=(16, 3, 32, 32))

# Rescale from [-1, 1] to [0, 1] for display
samples = (samples + 1.0) / 2.0

_, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
for i in range(16):
    ax = axes[i // 4][i % 4]
    img = samples[i].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)
    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
plt.suptitle("DDPM Generated Samples")
plt.tight_layout()
plt.show()
