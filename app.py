!pip install --upgrade pip

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Xtrafdme import estimate_sparse_blur 
from Xtrafdme import slic_process_image,snic_process_image

st.set_page_config(
    layout="wide",
)
# Title of the app
st.title("Image Input and Output Example")

# Sidebar or main page: Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    input_image = Image.open(uploaded_file)

    # Layout for side-by-side display with three images above and two below
    col1, es1, col2,es2, col3 = st.columns([1,0.5, 1,0.5, 1])  # For the first row (three columns)

    with col1:
        st.subheader("Original Image:")
        st.image(input_image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("Edges:")
        processed_image_1 = input_image.convert("L")
        image_np = np.array(processed_image_1)
        edges = cv2.Canny(image_np, threshold1=10, threshold2=50)
        edge_image = Image.fromarray(edges)
        st.image(edge_image, caption="Canny Edge Image", use_container_width=True)

    with col3:
        st.subheader("Sparse Defocus Map:")
        sparse_bmap,mag1,mag2 = estimate_sparse_blur(processed_image_1, edges, std1=1, std2=2)
        sparse_bmap = np.uint8(255 * (sparse_bmap - np.min(sparse_bmap)) / (np.max(sparse_bmap) - np.min(sparse_bmap)))
        st.image(sparse_bmap, caption="Sparse Defocus", use_container_width=True)


    # Second row with two columns
    es3,col4,es4, col5,es5 = st.columns([0.75,1,0.5, 1,0.75])  # For the second row (two columns)

    with col4:
        st.subheader("Dense Defocus:SLIC")
        seed_mask = sparse_bmap > 0
        dense_depth, superpixels = slic_process_image(
            np.asarray(input_image,dtype=np.float32), 
            sparse_bmap, 
            seed_mask,
            n_segments=200,
            compactness=20,
        )
        st.image(dense_depth, caption="Inverted Image", use_container_width=True)

    with col5:
        st.subheader("Dense Defocus:SNIC")
        seed_mask = sparse_bmap > 0
        dense_depth, superpixels = snic_process_image(
            np.asarray(input_image,dtype=np.float32),
            sparse_bmap, 
            seed_mask,
            n_segments=200,
            compactness=20,
        )
        st.image(dense_depth, caption="Processed Image 2", use_container_width=True)

