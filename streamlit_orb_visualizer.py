import cv2
import os
import numpy as np
import streamlit as st
from io import BytesIO

# --- UI SETTINGS ---
st.title("🔍 ORB Feature Matching Visualizer")
st.sidebar.header("⚙️ Parameters")

dataset_path = st.sidebar.text_input("Dataset Folder", "rgb")
rotation_angle = st.sidebar.slider("Rotation Angle (°)", 0, 90, 30)
n_features = st.sidebar.slider("Number of ORB Features", 100, 2000, 500)
good_match_ratio = st.sidebar.slider("Good Match Ratio", 0.5, 1.0, 0.75)
edge_thresh1 = st.sidebar.slider("Canny Threshold 1", 50, 300, 180)
edge_thresh2 = st.sidebar.slider("Canny Threshold 2", 50, 300, 200)
downscale = st.sidebar.slider("Downscale Factor", 0.25, 1.0, 1.0)

start = st.button("▶️ Run Visualization")

if start and os.path.isdir(dataset_path):
    images = sorted(os.listdir(dataset_path))
    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    prev_gray = None
    prev_des = None

    progress = st.progress(0)
    frame_placeholder = st.empty()
    match_placeholder = st.empty()

    for i, img_name in enumerate(images):
        img_path = os.path.join(dataset_path, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        if downscale != 1.0:
            frame = cv2.resize(frame, None, fx=downscale, fy=downscale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, edge_thresh1, edge_thresh2)

        kp, des = orb.detectAndCompute(gray, None)
        frame_with_kp = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Feature Matching
        if prev_gray is not None and prev_des is not None and des is not None:
            rows, cols = prev_gray.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
            rotated_prev = cv2.warpAffine(prev_gray, M, (cols, rows))
            kp_rot, des_rot = orb.detectAndCompute(rotated_prev, None)

            if des_rot is not None:
                knn_matches = bf.knnMatch(des_rot, des, k=2)
                good_matches = [m for m, n in knn_matches if m.distance < good_match_ratio * n.distance]
                match_img = cv2.drawMatches(rotated_prev, kp_rot, gray, kp, good_matches[:50], None, flags=2)

                # Convert BGR → RGB for Streamlit display
                match_placeholder.image(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB),
                                        caption=f"Good Matches ({len(good_matches)})")

        frame_placeholder.image(cv2.cvtColor(frame_with_kp, cv2.COLOR_BGR2RGB), caption=f"Frame: {img_name}")
        prev_gray = gray
        prev_des = des
        progress.progress((i+1)/len(images))

    st.success("✅ Visualization Complete")
else:
    st.info("Please provide a valid dataset path and click **Run Visualization**.")
