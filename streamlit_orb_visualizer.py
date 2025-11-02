# streamlit_orb_visualizer.py
import os
import time
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")
st.title("🔍 ORB Feature Matching Visualizer")

# --- Sidebar Parameters ---
st.sidebar.header("⚙️ Settings")
dataset_path = st.sidebar.text_input("Dataset Folder", "rgb")
rotation_angle = st.sidebar.slider("Rotation Angle (°)", 0, 90, 30)
n_features = st.sidebar.slider("Number of ORB Features", 100, 2000, 500)
good_match_ratio = st.sidebar.slider("Good Match Ratio", 0.5, 1.0, 0.75)
edge_thresh1 = st.sidebar.slider("Canny Threshold 1", 50, 300, 180)
edge_thresh2 = st.sidebar.slider("Canny Threshold 2", 50, 300, 200)
downscale = st.sidebar.slider("Downscale Factor", 0.25, 1.0, 1.0)
max_matches = st.sidebar.slider("Max Matches to Display", 10, 200, 50)

start = st.button("▶️ Run Visualization")

if start:
    if not os.path.isdir(dataset_path):
        st.error("❌ Dataset folder not found!")
    else:
        images = sorted(os.listdir(dataset_path))
        if len(images) == 0:
            st.warning("⚠️ Dataset folder is empty")
        else:
            # --- Initialize ORB and matcher ---
            orb = cv2.ORB_create(nfeatures=n_features)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)

            prev_gray = None
            prev_des = None

            # --- Placeholders for Streamlit ---
            frame_placeholder = st.empty()
            edge_placeholder = st.empty()
            match_placeholder = st.empty()
            progress = st.progress(0)

            for i, img_name in enumerate(images):
                img_path = os.path.join(dataset_path, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                # Optional downscaling
                if downscale != 1.0:
                    frame = cv2.resize(frame, None, fx=downscale, fy=downscale)

                start_time = time.time()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, edge_thresh1, edge_thresh2)

                # --- Keypoints ---
                kp, des = orb.detectAndCompute(gray, None)
                frame_with_kp = cv2.drawKeypoints(
                    frame, kp, None, color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )

                # --- Feature Matching with previous frame ---
                if prev_gray is not None and prev_des is not None and des is not None:
                    rows, cols = prev_gray.shape
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
                    rotated_prev = cv2.warpAffine(prev_gray, M, (cols, rows))
                    kp_rot, des_rot = orb.detectAndCompute(rotated_prev, None)

                    if des_rot is not None:
                        knn_matches = bf.knnMatch(des_rot, des, k=2)
                        good_matches = [m for m, n in knn_matches if m.distance < good_match_ratio * n.distance]

                        # Draw matches
                        match_img = cv2.drawMatches(
                            rotated_prev, kp_rot, gray, kp,
                            good_matches[:max_matches], None, flags=2
                        )

                        # Color-coded overlay
                        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        for m in good_matches[:max_matches]:
                            pt1 = tuple(map(int, kp_rot[m.queryIdx].pt))
                            pt2 = tuple(map(int, kp[m.trainIdx].pt))
                            color = (0, 255, 0) if m.distance < 40 else (0, 0, 255)
                            cv2.line(color_img, pt1, pt2, color, 1)

                        match_placeholder.image(
                            cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB),
                            caption=f"Good Matches ({len(good_matches)})",
                            use_column_width=True
                        )

                # --- Display keypoints and edges ---
                frame_placeholder.image(
                    cv2.cvtColor(frame_with_kp, cv2.COLOR_BGR2RGB),
                    caption=f"Frame: {img_name} — Keypoints: {len(kp)}",
                    use_column_width=True
                )
                edge_placeholder.image(edges, caption="Edges (Canny)", channels="GRAY", use_column_width=True)

                # --- Update previous frame ---
                prev_gray = gray
                prev_des = des

                # --- Update progress ---
                progress.progress((i+1)/len(images))

                # Optional: add small delay to see updates
                time.sleep(0.1)

            st.success("✅ Visualization Complete")
