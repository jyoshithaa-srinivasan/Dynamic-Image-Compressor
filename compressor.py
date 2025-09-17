# compressor.py
print(">>> compressor.py loaded from:", __file__)
from io import BytesIO
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

# TensorFlow for learned compression
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------
# Helper functions (PIL/JPEG)
# -------------------------
def build_soft_mask(size: Tuple[int, int], boxes: List[Tuple[int,int,int,int]],
                    feather: int = 15) -> Image.Image:
    """
    Returns an 'L' (grayscale) mask where ROIs are white (255) and background is black (0),
    with Gaussian blur feathering at edges.
    boxes: list of (x1,y1,x2,y2)
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
    Returns JPEG bytes with specified quality.
    subsampling=0 gives highest chroma fidelity; 1 or 2 increase compression.
    """
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=subsampling, optimize=optimize)
    return buf.getvalue()

def jpeg_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGB")

def composite_with_mask(hq_img: Image.Image, lq_img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Composites two same-size RGB images with a grayscale mask (L mode).
    White (255) takes from hq_img; black (0) takes from lq_img.
    """
    if hq_img.size != lq_img.size:
        lq_img = lq_img.resize(hq_img.size, Image.Resampling.LANCZOS)
    if mask.mode != "L":
        mask = mask.convert("L")
    return Image.composite(hq_img, lq_img, mask)

# -------------------------
# Original PIL-based selective_compress (unchanged)
# -------------------------
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
    Implements the two-pass ROI compression:
      1) Build soft mask from ROIs.
      2) Make HQ and LQ JPEG renders.
      3) Composite HQ and LQ by mask for a visually seamless result.
      4) Optionally re-encode the composite as JPEG to control final size.
    """
    img = base_img.convert("RGB")
    mask = build_soft_mask(img.size, rois, feather=feather)

    # Make JPEG renders
    hq_bytes = encode_jpeg_bytes(img, quality=q_fg, subsampling=subsampling_fg, optimize=True)
    lq_bytes = encode_jpeg_bytes(img, quality=q_bg, subsampling=subsampling_bg, optimize=True)
    hq_img = jpeg_from_bytes(hq_bytes)
    lq_img = jpeg_from_bytes(lq_bytes)

    blended = composite_with_mask(hq_img, lq_img, mask)

    # Optional final encode for size control
    if final_encode_quality is not None:
        final_bytes = encode_jpeg_bytes(blended, quality=int(final_encode_quality), subsampling=1, optimize=True)
        final_img = jpeg_from_bytes(final_bytes)
        size_bytes = len(final_bytes)
    else:
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

# -------------------------
# TensorFlow Autoencoder (learned) - loads weights if available
# -------------------------
def _build_autoencoder_tf(input_shape=(128,128,3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x)
    return models.Model(inputs, outputs)

# Try to load weights if file exists
_AUTOENCODER = None
_AUTOENCODER_PATH = os.path.join("models", "autoencoder.h5")
_AUTOENCODER_LOAD_ERROR = None
if os.path.exists(_AUTOENCODER_PATH):
    try:
        _AUTOENCODER = _build_autoencoder_tf()
        _AUTOENCODER.load_weights(_AUTOENCODER_PATH)
        # set to inference mode
        _AUTOENCODER.trainable = False
    except Exception as e:
        _AUTOENCODER = None
        _AUTOENCODER_LOAD_ERROR = str(e)
else:
    _AUTOENCODER = None
    _AUTOENCODER_LOAD_ERROR = f"Autoencoder weights not found at '{_AUTOENCODER_PATH}'. Train in Colab and place weights there."

def selective_compress_learned(
    base_img: Image.Image,
    rois: List[Tuple[int,int,int,int]],
    feather: int = 15,
    q_bg: int = 35,
    final_encode_quality: int | None = 85
) -> Dict:
    """
    Learned compression pipeline:
      - Reconstruct each ROI using a small TensorFlow autoencoder (128x128)
      - Encode background as low-quality JPEG
      - Composite reconstructed ROIs over the low-quality background with feathered mask
    Requires models/autoencoder_tf.h5 to be present (see README / training notes).
    Returns same dict shape as selective_compress for UI compatibility.
    """
    if _AUTOENCODER is None:
        raise RuntimeError("Autoencoder model not available. " + (_AUTOENCODER_LOAD_ERROR or ""))

    img = base_img.convert("RGB")
    w, h = img.size
    mask = build_soft_mask(img.size, rois, feather=feather)

    # Start HQ layer as a copy of the original, then replace ROI patches with autoencoder reconstructions
    hq_img = img.copy()

    for (x1, y1, x2, y2) in rois:
        # ensure integers & clamp to image bounds
        x1_i = max(0, int(round(x1)))
        y1_i = max(0, int(round(y1)))
        x2_i = min(w, int(round(x2)))
        y2_i = min(h, int(round(y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        roi = img.crop((x1_i, y1_i, x2_i, y2_i))
        roi_np = np.array(roi).astype(np.float32) / 255.0

        # Resize to autoencoder input (128x128)
        roi_resized = Image.fromarray((roi_np * 255).astype(np.uint8)).resize((128, 128), Image.Resampling.LANCZOS)
        roi_arr = np.array(roi_resized).astype(np.float32) / 255.0
        roi_batch = np.expand_dims(roi_arr, axis=0)  # shape (1,128,128,3)

        # Predict
        recon = _AUTOENCODER.predict(roi_batch, verbose=0)[0]
        recon_img = (np.clip(recon, 0.0, 1.0) * 255.0).astype(np.uint8)
        recon_pil = Image.fromarray(recon_img).resize((x2_i - x1_i, y2_i - y1_i), Image.Resampling.LANCZOS)

        # Paste reconstructed ROI into HQ image
        hq_img.paste(recon_pil, (x1_i, y1_i, x2_i, y2_i))

    # Create LQ background image via JPEG
    lq_bytes = encode_jpeg_bytes(img, quality=q_bg, subsampling=2, optimize=True)
    lq_img = jpeg_from_bytes(lq_bytes)

    # Composite with mask
    blended = composite_with_mask(hq_img, lq_img, mask)

    # Optional final encode
    if final_encode_quality is not None:
        final_bytes = encode_jpeg_bytes(blended, quality=int(final_encode_quality), subsampling=1, optimize=True)
        final_img = jpeg_from_bytes(final_bytes)
        size_bytes = len(final_bytes)
    else:
        final_img = blended
        size_bytes = None

    # prepare previews bytes (hq preview encoded as HQ JPEG for download/show)
    hq_bytes = encode_jpeg_bytes(hq_img, quality=95, subsampling=0, optimize=True)
    return {
        "result_img": final_img,
        "hq_preview": hq_img,
        "lq_preview": lq_img,
        "mask": mask,
        "size_bytes": size_bytes,
        "previews_bytes": (hq_bytes, lq_bytes)
    }
