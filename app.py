#streamlit ui main entry point
import streamlit as st
from detector import detect_objects
from PIL import Image, ImageDraw

st.title("Dynamic Content Aware Image Compression")

# Step 1: upload img
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

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

