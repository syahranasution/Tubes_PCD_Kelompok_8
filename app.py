import os

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Load model KNN dan Label Encoder
try:
    knn = joblib.load("knn_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error(
        "File (knn_model.pkl atau label_encoder.pkl) tidak ditemukan. Pastikan model sudah dilatih dan disimpan di folder yang sama."
    )
    st.stop()


# Segmentasi dengan K-Means Clustering
def segmentation_kmeans_hsv(image):
    h, w, _ = image.shape
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # K-Means parameters
    k_clusters = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, labels_kmeans, centers = cv2.kmeans(
        pixels, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    best_cluster_label = -1
    max_score = -1

    # Cari cluster terbaik (heuristik: saturasi rata-rata dan jumlah piksel)
    for i in range(k_clusters):
        cluster_mask = (labels_kmeans == i).reshape(h, w).astype(np.uint8) * 255
        hsv_pixels_in_cluster = image[cluster_mask > 0]

        # Hitung saturasi rata-rata kluster
        if len(hsv_pixels_in_cluster) > 0:
            avg_saturation = np.mean(hsv_pixels_in_cluster[:, 1])
            # Saturation > 5
            colored_pixels_count = np.sum(hsv_pixels_in_cluster[:, 1] > 5)

            current_score = avg_saturation * colored_pixels_count
            if current_score > max_score:
                max_score = current_score
                best_cluster_label = i

    # Tidak ada cluster yang bisa digunakan
    if best_cluster_label == -1:
        return None

    # Buat mask dari cluster terbaik
    product_mask = (labels_kmeans == best_cluster_label).reshape(h, w).astype(
        np.uint8
    ) * 255

    masked_image = cv2.bitwise_and(image, image, mask=product_mask)

    # Hanya gunakan segmentasi jika hasilnya tidak kosong
    if masked_image.shape[0] == 0:
        return None

    return product_mask


# Fungsi untuk mengekstrak 3 fitur (HSV)
def extract_dominant_color_hsv(image):
    if image is None:
        return None

    # Konversi gambar ke HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Segmentasi
    segmentation_mask = segmentation_kmeans_hsv(hsv_image)

    if segmentation_mask is None:
        return None

    # Hitung histogram
    # Gunakan bin 18, 16, 16
    hist = cv2.calcHist(
        [hsv_image],
        [0, 1, 2],
        segmentation_mask,
        [18, 16, 16],
        [0, 180, 0, 256, 0, 256],
    )

    # Normalisasi histogram
    hist /= hist.sum()

    # Temukan warna dominan
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    hue = max_idx[0] * (180 // hist.shape[0])  # 180 / 18 = 10
    saturation = max_idx[1] * (256 // hist.shape[1])  # 256 / 16 = 16
    value = max_idx[2] * (256 // hist.shape[2])  # 256 / 16 = 16

    return hue, saturation, value


# UI Streamlit
st.set_page_config(page_title="Klasifikasi Warna Produk dengan KNN", layout="centered")

st.title("Klasifikasi Warna Produk dengan KNN")
st.write("Upload gambar produk untuk mengklasifikasikan warnanya.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")

    # Convert image ke format OpenCV
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Ekstrasi fitur warna dominan
    color_features = extract_dominant_color_hsv(opencv_image)

    if color_features is not None:
        img_array = np.array(color_features).reshape(1, -1)

        # Prediksi dengan model KNN yang sudah dilatih
        prediction_encoded = knn.predict(img_array)
        predicted_label = label_encoder.inverse_transform(prediction_encoded)[0]

        st.success(f"**Prediksi Warna:** {predicted_label}")
    else:
        st.warning(
            "Terjadi kesalahan saat mengekstrak fitur warna. Silahkan coba gambar lain."
        )
