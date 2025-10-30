# enhancer.py
import os
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "enhancer.pth")
_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage()

# ----- small generator architecture (same as earlier) -----
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
        )
    def forward(self, x): return x + self.net(x)

class EnhancerGenerator(nn.Module):
    def __init__(self, in_ch=4, base=64, n_res=4):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 7, padding=3), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 4, 2, 1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base*2, base*4, 4, 2, 1), nn.ReLU(True))
        self.res = nn.Sequential(*[ResidualBlock(base*4) for _ in range(n_res)])
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base*2, base, 4, 2, 1), nn.ReLU(True))
        self.dec1 = nn.Conv2d(base, 3, 7, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, jpeg, mask):
        x = torch.cat([jpeg, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        r = self.res(e3)
        d3 = self.dec3(r)
        d2 = self.dec2(d3 + e2)
        out = self.dec1(d2 + e1)
        out = (self.tanh(out) + 1.0) / 2.0
        return out

# ----- loader and inference helper -----
def _prepare_inputs(pil_img, pil_mask, device=DEVICE):
    img_t = _to_tensor(pil_img).unsqueeze(0).to(device)
    mask_t = _to_tensor(pil_mask.convert("L")).unsqueeze(0).to(device)
    if mask_t.shape[1] > 1:
        mask_t = mask_t[:, :1, :, :]
    return img_t, mask_t

def load_generator(path=None, device=DEVICE):
    path = path or DEFAULT_MODEL_PATH
    gen = EnhancerGenerator().to(device)
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location=device)
            state = ckpt.get("generator", ckpt)
            gen.load_state_dict(state)
            gen.eval()
            print(f"[enhancer] loaded generator from {path}")
            return gen, True
        except Exception as e:
            print(f"[enhancer] failed to load checkpoint: {e} — using demo fallback")
    return gen, False

def _pil_unsharp_enhance(pil_img, radius=2, percent=150, threshold=3):
    return pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

@torch.no_grad()
def enhance_and_blend(jpeg_pil: Image.Image, mask_pil: Image.Image,
                      model_path: str | None = None, blend: bool = True, return_debug: bool = False):
    """
    Run enhancer on jpeg_pil using mask_pil.
    If a trained model (enhancer.pth) exists it will be used.
    Otherwise a lightweight PIL unsharp filter is used (demo mode).
    If return_debug True, returns dict with intermediates: 'lq','enhanced','out','mask','diff_heat'.
    """
    gen, model_exists = load_generator(model_path, device=DEVICE)
    if model_exists:
        img_t, mask_t = _prepare_inputs(jpeg_pil, mask_pil)
        fake_t = gen(img_t, mask_t).clamp(0,1)
        fake_pil = _to_pil(fake_t[0].cpu())
        if blend:
            jpeg_np = np.array(jpeg_pil).astype(np.float32) / 255.0
            fake_np = np.array(fake_pil).astype(np.float32) / 255.0
            mask_np = (np.array(mask_pil.convert("L")).astype(np.float32) / 255.0)[..., None]
            out_np = jpeg_np * (1.0 - mask_np) + fake_np * mask_np
            out_pil = Image.fromarray((out_np * 255).astype("uint8"))
        else:
            out_pil = fake_pil
        if return_debug:
            diff = (np.abs(np.array(fake_pil).astype(int) - np.array(jpeg_pil).astype(int))).astype(np.uint8)
            diff_gray = np.mean(diff, axis=2).astype(np.uint8)
            heat = Image.fromarray(diff_gray).convert("L").resize(jpeg_pil.size)
            return {"lq": jpeg_pil, "enhanced": fake_pil, "out": out_pil, "mask": mask_pil, "diff_heat": heat}
        return out_pil

    # DEMO fallback (no trained model): simple unsharp inside mask
    enhanced_pil = _pil_unsharp_enhance(jpeg_pil)
    jpeg_np = np.array(jpeg_pil).astype(np.float32) / 255.0
    enhanced_np = np.array(enhanced_pil).astype(np.float32) / 255.0
    mask_np = (np.array(mask_pil.convert("L")).astype(np.float32) / 255.0)[..., None]
    out_np = jpeg_np * (1.0 - mask_np) + enhanced_np * mask_np
    out_pil = Image.fromarray((out_np * 255).astype("uint8"))
    if return_debug:
        diff = (np.abs(np.array(enhanced_pil).astype(int) - np.array(jpeg_pil).astype(int))).astype(np.uint8)
        diff_gray = np.mean(diff, axis=2).astype(np.uint8)
        heat = Image.fromarray(diff_gray).convert("L").resize(jpeg_pil.size)
        return {"lq": jpeg_pil, "enhanced": enhanced_pil, "out": out_pil, "mask": mask_pil, "diff_heat": heat}
    return out_pil
