# compressor.py
print(">>> compressor.py loaded from:", __file__)
from io import BytesIO
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

def build_soft_mask(size: Tuple[int, int], boxes: List[Tuple[int,int,int,int]],
                    feather: int = 15) -> Image.Image:
    """
    Returns an 'L' (grayscale) mask where ROIs are white (255) and background is black (0),
    with Gaussian blur feathering at edges.
    """
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask

def encode_jpeg_bytes(img: Image.Image, quality: int, subsampling: int = 1, optimize: bool = True) -> bytes:
    """
    Returns JPEG bytes with specified quality. subsampling=0 gives highest chroma fidelity;
    1 or 2 increase compression. Optimize may slightly reduce size by optimizing Huffman tables.
    """
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=subsampling, optimize=optimize)
    return buf.getvalue()

def jpeg_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGB")

def composite_with_mask(hq_img: Image.Image, lq_img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Composites two same‑size RGB images with a grayscale mask (L mode).
    White (255) takes from hq_img; black (0) takes from lq_img.
    """
    if hq_img.size != lq_img.size:
        lq_img = lq_img.resize(hq_img.size, Image.Resampling.LANCZOS)
    if mask.mode != "L":
        mask = mask.convert("L")
    return Image.composite(hq_img, lq_img, mask)

def selective_compress(
    base_img: Image.Image,
    rois: List[Tuple[int,int,int,int]],
    q_fg: int = 90,
    q_bg: int = 35,
    feather: int = 15,
    subsampling_fg: int = 0,
    subsampling_bg: int = 2,
    final_encode_quality: int | None = 85
) -> Dict:
    """
    Implements the two‑pass ROI compression:
      1) Build soft mask from ROIs.
      2) Make HQ and LQ JPEG renders.
      3) Composite HQ and LQ by mask for a visually seamless result.
      4) Optionally re‑encode the composite as JPEG to control final size.

    Returns dict with:
      - result_img (PIL Image)
      - hq_preview (PIL Image)
      - lq_preview (PIL Image)
      - mask (PIL L image)
      - size_bytes (int)
      - previews_bytes (tuple)
    """
    img = base_img.convert("RGB")
    mask = build_soft_mask(img.size, rois, feather=feather)

    # Make JPEG renders
    hq_bytes = encode_jpeg_bytes(img, quality=q_fg, subsampling=subsampling_fg, optimize=True)
    lq_bytes = encode_jpeg_bytes(img, quality=q_bg, subsampling=subsampling_bg, optimize=True)
    hq_img = jpeg_from_bytes(hq_bytes)
    lq_img = jpeg_from_bytes(lq_bytes)

    # Composite with mask
    blended = composite_with_mask(hq_img, lq_img, mask)

    # Optional final encode for size control
    if final_encode_quality is not None:
        final_bytes = encode_jpeg_bytes(blended, quality=int(final_encode_quality), subsampling=1, optimize=True)
        final_img = jpeg_from_bytes(final_bytes)
        size_bytes = len(final_bytes)
    else:
        # Keep raw blended RGB (PNG on download would be large, so recommend final_encode_quality)
        final_img = blended
        size_bytes = None

    return {
        "result_img": final_img,
        "hq_preview": hq_img,
        "lq_preview": lq_img,
        "mask": mask,
        "size_bytes": size_bytes,
        "previews_bytes": (hq_bytes, lq_bytes)
    }
