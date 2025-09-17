# app.py
import streamlit as st
from detector import detect_objects
from PIL import Image, ImageDraw
from io import BytesIO
from compressor import selective_compress, selective_compress_learned
from streamlit_image_comparison import image_comparison
import cv2
import numpy as np

st.set_page_config(layout="wide")
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

    # allow multiple selections
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

        # Preset choices (keep same UX as your original)
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
            hq_quality_default, lq_quality_default, feather_default = 85, 40, 20
        elif preset == "Sharper Faces & Text":
            hq_quality_default, lq_quality_default, feather_default = 95, 50, 15
        else:  # Maximum Size Reduction
            hq_quality_default, lq_quality_default, feather_default = 80, 25, 30

        # Optional advanced settings (kept exactly as in your original project)
        with st.expander("⚙️ Advanced Settings"):
            hq_quality = st.slider("High Quality Area", 70, 100, hq_quality_default)
            lq_quality = st.slider("Background Quality", 10, 80, lq_quality_default)
            feather = st.slider("Blend Edge Smoothness", 5, 50, feather_default)

        # ------------------ Run Compression Buttons ------------------
        # Compute selected boxes
        selected_boxes = [det["bbox"] for det, opt in zip(detections, options) if opt in selected_ids]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Compress Image (JPEG)"):
                # call your existing PIL/JPEG-based pipeline
                out = selective_compress(
                    st.session_state["original_img"],
                    selected_boxes,
                    q_fg=hq_quality,
                    q_bg=lq_quality,
                    feather=feather
                )
                st.session_state["compression_out"] = out
                st.session_state["compression_params"] = {
                    "method": "jpeg",
                    "hq_q": hq_quality,
                    "lq_q": lq_quality,
                    "feather": feather
                }

        with col2:
            if st.button("Compress Image (Autoencoder)"):
                # call learned compression pipeline (TensorFlow)
                try:
                    out = selective_compress_learned(
                        st.session_state["original_img"],
                        selected_boxes,
                        feather=feather,
                        q_bg=lq_quality,
                        final_encode_quality=85
                    )
                    st.session_state["compression_out"] = out
                    st.session_state["compression_params"] = {
                        "method": "autoencoder",
                        "feather": feather
                    }
                except Exception as e:
                    st.error(f"Autoencoder compression failed: {e}")
                    st.info("If you haven't trained/downloaded the autoencoder weights, train in Colab and place the file at models/autoencoder_tf.h5")

# ------------------ Results & Download ------------------
if "compression_out" in st.session_state:
    out = st.session_state["compression_out"]
    params = st.session_state.get("compression_params", {})

    st.subheader("Preview: Before vs After")

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

    st.download_button("⬇️ Download Compressed JPEG", data=data_bytes,
                       file_name="compressed.jpg", mime="image/jpeg")
