# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd
from gradcam_new import GradCAM
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Skin Type Classifier - MobileNetV2 & Grad-CAM",
    layout="wide"
)

# ==================== CONSTANT ====================
CLASS_NAMES = ['dry', 'normal', 'oily', 'sensitive', 'combination']
CLASS_NAMES_ID = ['Dry (Kering)', 'Normal', 'Oily (Berminyak)', 
                  'Sensitive (Sensitif)', 'Combination (Kombinasi)']

IMG_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    print("LOADING MODEL FINAL")
    return tf.keras.models.load_model(
        'models/mobilenetv2_5classes_final.h5'
    )

    for p in paths:
        if os.path.exists(p):
            print(f"Loading model: {p}")
            return tf.keras.models.load_model(p)

    return None

# ==================== PREPROCESS ====================
def preprocess(uploaded_file):

    img = Image.open(
        uploaded_file
    ).convert('RGB')

    original = img.copy()

    # resize langsung
    img = img.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # normalisasi
    arr = np.array(img)/255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    return original, arr

# ==================== PREDICT ====================
def predict(model, img_array):
    pred = model.predict(img_array, verbose=0)[0]

    print("\n===================")
    print("Probabilitas Prediksi:")
    print("Dry        :", pred[0])
    print("Normal     :", pred[1])
    print("Oily       :", pred[2])
    print("Sensitive  :", pred[3])
    print("Combination:", pred[4])
    print("===================\n")

    idx = np.argmax(pred)
    conf = pred[idx]

    results = []
    for i in range(len(CLASS_NAMES)):
        results.append({
            "class": CLASS_NAMES_ID[i],
            "confidence": float(pred[i])
        })

    results.sort(
        key=lambda x: x['confidence'],
        reverse=True
    )

    return idx, conf, results

# ==================== GRADCAM HELPER ====================
def generate_gradcam(model, img_array, class_idx, original_img):
    gradcam = GradCAM(model, class_idx)

    heatmap = gradcam.compute_heatmap(img_array)

    # Normalisasi
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Resize ke ukuran gambar asli
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))

    # Convert ke 0–255
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    original = np.array(original_img)
    superimposed = cv2.addWeighted(heatmap_color, 0.4, original, 0.6, 0)

    return heatmap_color, superimposed

# ==================== MAIN ====================
def main():
    st.title("Klasifikasi Jenis Kulit Wajah Menggunakan MobileNetV2 dan Grad-CAM")

    st.markdown("""
     
    """)

    model = load_model()

    if model is None:
        st.error("Model tidak ditemukan.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Unggah citra wajah",
        type=['jpg','png','jpeg']
    )

    if uploaded_file:

        # ==================== INPUT & PREDIKSI ====================

        col1, col2 = st.columns(2)

        with col1:
            original, img_array = preprocess(uploaded_file)

            st.image(
                original,
                caption="Citra Input",
                use_container_width=True
            )

        with col2:

            idx, conf, results = predict(
                model,
                img_array
            )

            st.subheader("Hasil Prediksi")

            st.write(
                f"Kelas: {CLASS_NAMES_ID[idx]}"
            )

            st.write(
                f"Tingkat Kepercayaan: {conf:.2%}"
            )

            df = pd.DataFrame(results)

            st.dataframe(df)

        # ==================== GRADCAM ====================

        st.subheader(
            "Visualisasi Grad-CAM"
        )

        heatmap, superimposed = generate_gradcam(
            model,
            img_array,
            idx,
            original
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.image(
                original,
                caption="Citra Asli",
                use_container_width=True
            )

        with col_b:
            st.image(
                heatmap,
                caption="Heatmap Grad-CAM",
                use_container_width=True
            )

        with col_c:
            st.image(
                superimposed,
                caption="Superimposed",
                use_container_width=True
            )

        # ==================== ANALISIS ====================

        st.subheader(
            "Analisis Interpretasi Grad-CAM"
        )

        if CLASS_NAMES[idx] == "oily":

            st.write("""
            Area aktivasi tinggi pada heatmap
            menunjukkan fokus model pada area
            wajah tertentu seperti dahi,
            hidung, atau area T-zone
            yang berkaitan dengan
            produksi minyak.
            """)

        elif CLASS_NAMES[idx] == "dry":

            st.write("""
            Aktivasi model terlihat
            pada area kulit tertentu
            yang berkaitan dengan
            karakteristik kulit kering.
            """)

        elif CLASS_NAMES[idx] == "sensitive":

            st.write("""
            Aktivasi tinggi muncul
            pada area kulit tertentu
            yang menunjukkan
            karakteristik visual
            berkaitan dengan
            kulit sensitif.
            """)

        elif CLASS_NAMES[idx] == "combination":

            st.write("""
            Model memperhatikan
            lebih dari satu area wajah
            yang menunjukkan kombinasi
            karakteristik kulit.
            """)

        else:

            st.write("""
            Aktivasi model tersebar
            lebih merata pada area wajah
            yang menunjukkan karakteristik
            kulit normal.
            """)

        # ==================== EVALUASI ====================

        st.subheader(
            "Evaluasi Heatmap"
        )

        st.info("""
        Heatmap dianggap baik apabila
        area merah atau kuning
        berfokus pada bagian kulit wajah
        seperti dahi, pipi,
        hidung, atau T-zone.

        Heatmap dianggap kurang baik
        apabila aktivasi tinggi muncul
        pada rambut, background,
        pakaian, atau objek lain
        di luar area kulit wajah.
        """)

        # ==================== KETERANGAN ====================

        st.markdown("""
        **Keterangan Visualisasi Grad-CAM:**

        Grad-CAM digunakan untuk
        mengidentifikasi area penting
        pada citra yang berkontribusi
        terhadap hasil prediksi model.

        - Area merah menunjukkan kontribusi tinggi
        - Area biru menunjukkan kontribusi rendah
        - Superimposed merupakan gabungan citra asli dan heatmap

        Visualisasi membantu memahami
        fokus model dalam melakukan klasifikasi.
        """)


if __name__ == "__main__":
    main()