#streamlit ui main entry point
import streamlit as st
from detector import detect_objects
from PIL import Image, ImageDraw
import json
from io import BytesIO
from compressor import selective_compress
from streamlit_image_comparison import image_comparison
import cv2
import numpy as np
import os
import uuid
from enhancer import enhance_and_blend, DEFAULT_MODEL_PATH
st.title("Dynamic Content Aware Image Compression")

# Step 1: upload img
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if "original_img" not in st.session_state:
        st.session_state["original_img"] = image
    # Step 2: Run object detection
    if st.button("Run Object Detection"):
        detections, annotated_img = detect_objects(image)
        st.session_state["detections"] = detections  # save to session
        st.session_state["original_img"] = image
        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

# Step 3: User selection for ROI (persistent widget)
if "detections" in st.session_state:
    st.subheader("Select Objects to Preserve in High Quality")

    detections = st.session_state["detections"]
    options = [f"{det['label']} (conf: {det['confidence']:.2f})" for det in detections]

    selected_ids = st.multiselect("Choose detected objects:", options)

    if selected_ids:
        # Copy original image
        img_copy = st.session_state["original_img"].copy()
        draw = ImageDraw.Draw(img_copy)

        # Draw bounding boxes only for selected objects
        for det, option in zip(detections, options):
            if option in selected_ids:
                x1, y1, x2, y2 = det["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
                draw.text((x1, y1 - 10), det["label"], fill="lime")

        st.image(img_copy, caption="Selected Objects Highlighted", use_column_width=True)
        st.subheader("Compression Settings")

preset = st.radio(
    "Choose a mode:",
    [
        "Balanced Compression (recommended)",
        "Sharper Faces & Text",
        "Maximum Size Reduction"
    ],
    index=0
)

# Preset values
if preset == "Balanced Compression (recommended)":
    hq_quality, lq_quality, feather = 85, 40, 20
elif preset == "Sharper Faces & Text":
    hq_quality, lq_quality, feather = 95, 50, 15
else:  # Maximum Size Reduction
    hq_quality, lq_quality, feather = 80, 25, 30

# Optional advanced settings
with st.expander("⚙️ Advanced Settings"):
    hq_quality = st.slider("High Quality Area", 70, 100, hq_quality)
    lq_quality = st.slider("Background Quality", 10, 80, lq_quality)
    feather = st.slider("Blend Edge Smoothness", 5, 50, feather)

# ------------------ Compression Function ------------------
def selective_compress(image, hq_q, lq_q, feather):
    # Convert PIL → OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Simulate detection mask (for demo, central region)
    mask = np.zeros(img_cv.shape[:2], np.uint8)
    h, w = mask.shape
    cv2.rectangle(mask, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), 255, -1)
    mask = cv2.GaussianBlur(mask, (feather|1, feather|1), 0)

    # Encode HQ and LQ layers
    _, hq_buf = cv2.imencode(".jpg", img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), hq_q])
    _, lq_buf = cv2.imencode(".jpg", img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), lq_q])
    hq_img = cv2.imdecode(hq_buf, 1)
    lq_img = cv2.imdecode(lq_buf, 1)

    # Blend by mask
    mask_f = mask.astype(float) / 255.0
    mask_f = cv2.merge([mask_f]*3)
    blended = (hq_img * mask_f + lq_img * (1-mask_f)).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)), \
           Image.fromarray(mask), \
           Image.fromarray(cv2.cvtColor(hq_img, cv2.COLOR_BGR2RGB)), \
           Image.fromarray(cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB))

# ------------------ Run Compression ------------------
if uploaded_file and st.button("Compress Image"):
    result_img, mask, hq_preview, lq_preview = selective_compress(
        st.session_state["original_img"], hq_quality, lq_quality, feather
    )
    st.session_state["compression_out"] = {
        "result_img": result_img,
        "mask": mask,
        "hq_preview": hq_preview,
        "lq_preview": lq_preview
    }
    st.session_state["compression_params"] = {
        "hq_q": hq_quality,
        "lq_q": lq_quality,
        "feather": feather
    }
# ------------------ Results & Download ------------------
if "compression_out" in st.session_state:
    out = st.session_state["compression_out"]
    params = st.session_state["compression_params"]

    st.subheader("Preview: Before vs After")

    # Before/After slider
    image_comparison(
        img1=st.session_state["original_img"],
        img2=out["result_img"],
        label1="Original",
        label2="Compressed",
        width=700
    )

    st.subheader("Details")
    c1, c2, c3 = st.columns(3)
    with c1: st.image(out["hq_preview"], caption="High Quality Layer", use_container_width=True)
    with c2: st.image(out["lq_preview"], caption="Compressed Background", use_container_width=True)
    with c3: st.image(out["mask"], caption="Object Mask", use_container_width=True)

    # File size comparison
    buf = BytesIO()
    out["result_img"].save(buf, format="JPEG", quality=85, subsampling=1, optimize=True)
    data_bytes = buf.getvalue()

    orig_buf = BytesIO()
    st.session_state["original_img"].save(orig_buf, format="JPEG", quality=95)
    orig_size = len(orig_buf.getvalue())
    new_size = len(data_bytes)

    st.success(f"✅ Size reduced from {orig_size/1024:.1f} KB → {new_size/1024:.1f} KB "
               f"({100*(1 - new_size/orig_size):.1f}% smaller)")

    st.download_button(
        "⬇️ Download Compressed JPEG",
        data=data_bytes,
        file_name=f"compressed_{uuid.uuid4().hex[:8]}.jpg",
        mime="image/jpeg",
        key=f"download_compressed_{uuid.uuid4().hex}"
    )
    st.markdown("---")
    st.subheader("Optional: Enhance Compressed Image with GAN")

    # unique key to avoid duplicate widget ids
    apply_gan = st.checkbox(
        "Apply GAN Enhancement (use model if present)",
        value=False,
        key="apply_gan_after_slider"
    )
    enhanced_img = None

    if apply_gan and "compression_out" in st.session_state:
        out = st.session_state["compression_out"]
        mask = out["mask"]
        lq_preview = out["lq_preview"]
        model_path = DEFAULT_MODEL_PATH
        st.info(f"Running enhancer (model path: {model_path})")

        try:
            debug = enhance_and_blend(lq_preview, mask, model_path=model_path, blend=True, return_debug=True)
            enhanced_img = debug["out"]
            st.subheader("GAN Enhanced Result")
            st.image(enhanced_img, use_column_width=True)
            st.subheader("Per-pixel difference (gray heatmap)")
            st.image(debug["diff_heat"], use_column_width=True)
        except Exception as e:
            st.error(f"Enhancer failed: {e}")

    if enhanced_img is not None:
        st.session_state["compression_out"]["result_enhanced"] = enhanced_img

        # download enhanced image (unique key)
        buf_en = BytesIO()
        enhanced_img.save(buf_en, format="JPEG", quality=85, optimize=True)
        st.download_button(
            "⬇️ Download Enhanced JPEG",
            data=buf_en.getvalue(),
            file_name=f"enhanced_{uuid.uuid4().hex[:8]}.jpg",
            mime="image/jpeg",
            key=f"download_enh_{uuid.uuid4().hex}"
        )

        # enhanced vs compressed comparison
        st.subheader("Enhanced vs Compressed Comparison")
        image_comparison(
            img1=out["result_img"],
            img2=enhanced_img,
            label1="Compressed",
            label2="GAN Enhanced",
            width=700
        )