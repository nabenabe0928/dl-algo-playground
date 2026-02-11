import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from cifar10.dataset import get_data_loaders
from transformer.model import SimpleViT

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18


def plot_attn_map(ax: plt.Axes, img: torch.Tensor, attn_map: torch.Tensor) -> None:
    grid_size = 8
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_resized = (
        F.interpolate(
            attn_map[None, None],
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    img_ = img.permute(1, 2, 0).cpu().numpy()
    img_ = (img_ - img_.min()) / (img_.max() - img_.min())
    ax.imshow(img_)
    ax.imshow(attn_resized, cmap="jet", alpha=0.2)
    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)


batch_size = 128
_, val_loader = get_data_loaders(batch_size=batch_size)
device = torch.device("cpu")
model = SimpleViT()
model.load_state_dict(torch.load("models/vit_cifar10_50epochs.pth", map_location=device))
model.eval()
model.blocks[-1].attn.save_attn = True
shape = (batch_size, 8, 8)
_, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        model(imgs)
        attn_map = model.blocks[-1].attn.last_attn_weights.mean(axis=1)[:, 0, 1:].reshape(*shape)
        break

for i, img_id in enumerate(range(4)):
    ax = axes[i // 2][i % 2]
    ax.set_title(classes[labels[img_id]])
    plot_attn_map(ax, imgs[img_id], attn_map[img_id])
plt.show()
