
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 224

# Base pipeline: raw PIL image -> common 224x224 RGB float tensor in [0, 1]
_to_common = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),  # -> [0, 1], C x H x W
])


def to_common_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _to_common(img)


# Per-backbone normalization (applied AFTER all common-space interventions)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def normalize_clip(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# Color interventions
def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    return gray.repeat(1, 3, 1, 1)


def hue_rotate(x: torch.Tensor, degrees: float = 90.0) -> torch.Tensor:
    import matplotlib.colors as mcolors

    b = x.shape[0]
    x_np = x.permute(0, 2, 3, 1).detach().cpu().numpy()  # B,H,W,3
    out = np.empty_like(x_np)
    shift = (degrees % 360.0) / 360.0
    for i in range(b):
        hsv_img = mcolors.rgb_to_hsv(np.clip(x_np[i], 0.0, 1.0))
        hsv_img[..., 0] = (hsv_img[..., 0] + shift) % 1.0
        out[i] = mcolors.hsv_to_rgb(hsv_img)
    out_t = torch.from_numpy(out).permute(0, 3, 1, 2).float().to(x.device)
    return out_t.clamp(0, 1)



# Translation
def translate_image(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return x
    pad = max(abs(dx), abs(dy))
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    h, w = x.shape[-2:]
    top = pad - dy
    left = pad - dx
    return x_pad[:, :, top:top + h, left:left + w]


# (dx_sign, dy_sign) for right, left, down, up
_CARDINAL_DIRECTIONS = [("right", (1, 0)), ("left", (-1, 0)),
                         ("down", (0, 1)), ("up", (0, -1))]


def translate_all_directions(x: torch.Tensor, delta: int):
    if delta == 0:
        yield "none", x
        return
    for name, (sx, sy) in _CARDINAL_DIRECTIONS:
        yield name, translate_image(x, sx * delta, sy * delta)


# Patch structure (grid shuffle)
def patch_shuffle(x: torch.Tensor, grid: int = 4, seed: int = 6304) -> torch.Tensor:
    b, c, h, w = x.shape
    assert h % grid == 0 and w % grid == 0, "image size must be divisible by grid"
    ph, pw = h // grid, w // grid
    n_patches = grid * grid
    out = torch.empty_like(x)

    for i in range(b):
        gen = torch.Generator().manual_seed(seed + i)  # deterministic per image index
        perm = torch.randperm(n_patches, generator=gen)
        identity = torch.arange(n_patches)
        while torch.equal(perm, identity):
            perm = torch.randperm(n_patches, generator=gen)

        patches = x[i].unfold(1, ph, ph).unfold(2, pw, pw)  # C, grid, grid, ph, pw
        patches = patches.contiguous().view(c, n_patches, ph, pw)
        shuffled = patches[:, perm, :, :]
        shuffled = shuffled.view(c, grid, grid, ph, pw)
        shuffled = shuffled.permute(0, 1, 3, 2, 4).contiguous().view(c, h, w)
        out[i] = shuffled

    return out


