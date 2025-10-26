import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import gdown

# =========================
# Load model dan label (versi Google Drive)
# =========================
MODEL_PATH = "durian_leaf_disease_model.h5"
CLASS_INDEX_PATH = "class_indices.json"

# 🔽 Ganti ID ini dengan ID file Drive kamu
drive_url = "https://huggingface.co/Syefara/durian-leaf-disease/resolve/main/durian_leaf_disease_model.h5"  # <-- ganti sesuai ID file kamu

# Jika model belum ada di lokal, unduh otomatis
if not os.path.exists(MODEL_PATH):
    with st.spinner("📦 Mengunduh model dari Hugging Face..."):
        r = requests.get(model_url)
        open(MODEL_PATH, "wb").write(r.content)

# Load model setelah diunduh
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

# Membalik mapping agar index -> nama kelas
class_labels = {v: k for k, v in class_indices.items()}


with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

# Membalik mapping agar index -> nama kelas
class_labels = {v: k for k, v in class_indices.items()}

# =========================
# Informasi penyakit
# =========================
disease_info = {
    "ALGAL_LEAF_SPOT": {
        "Deskripsi": "Penyakit ini disebabkan oleh ganggang hijau parasit (Cephaleuros virescens) yang tumbuh di permukaan daun.",
        "Gejala": [
            "Bercak hijau keabu-abuan pada permukaan daun",
            "Permukaan daun terasa kasar",
            "Pertumbuhan daun terhambat"
        ],
        "Penyebab": [
            "Kelembapan tinggi dan ventilasi udara buruk",
            "Kebersihan kebun yang kurang terjaga"
        ],
        "Saran": [
            "Gunakan fungisida berbahan tembaga secara berkala",
            "Pangkas daun yang terinfeksi",
            "Jaga sirkulasi udara antar tanaman"
        ]
    },
    "ALLOCARIDARA_ATTACK": {
        "Deskripsi": "Serangan hama Allocaridara (sejenis serangga penghisap) yang menyebabkan daun rusak.",
        "Gejala": [
            "Bercak kuning tidak beraturan pada daun",
            "Daun menggulung dan mengering",
            "Terdapat serangga kecil di bawah permukaan daun"
        ],
        "Penyebab": [
            "Serangan hama Allocaridara sp.",
            "Kebersihan kebun yang buruk"
        ],
        "Saran": [
            "Gunakan insektisida sistemik sesuai dosis",
            "Lakukan monitoring populasi hama rutin",
            "Pelihara predator alami seperti laba-laba atau kepik"
        ]
    },
    "HEALTHY_LEAF": {
        "Deskripsi": "Daun durian dalam kondisi sehat tanpa tanda-tanda penyakit atau serangan hama.",
        "Gejala": [
            "Permukaan daun hijau mengkilap",
            "Tidak ada bercak atau luka"
        ],
        "Penyebab": [
            "Tanaman tumbuh optimal dengan nutrisi seimbang"
        ],
        "Saran": [
            "Pertahankan pola pemupukan yang seimbang",
            "Jaga kebersihan kebun dan sistem drainase",
            "Pantau kondisi tanaman secara berkala"
        ]
    },
    "LEAF_BLIGHT": {
        "Deskripsi": "Penyakit bercak daun yang disebabkan oleh jamur (Cercospora atau Alternaria) yang dapat melemahkan tanaman.",
        "Gejala": [
            "Bercak kecil berwarna coklat atau hitam pada daun",
            "Bercak dikelilingi halo kuning",
            "Daun berlubang dan rontok jika infeksi berat"
        ],
        "Penyebab": [
            "Infeksi jamur pada kondisi lembap",
            "Kelembapan tinggi dan percikan air hujan"
        ],
        "Saran": [
            "Gunakan fungisida sesuai penyebab",
            "Buang daun yang terinfeksi berat",
            "Perbaiki drainase dan hindari percikan air ke daun"
        ]
    },
    "PHOMOPSIS_LEAF_SPOT": {
        "Deskripsi": "Penyakit bercak daun akibat infeksi jamur Phomopsis sp.",
        "Gejala": [
            "Bercak kecil coklat pada daun tua",
            "Pinggiran bercak berwarna lebih gelap",
            "Daun gugur prematur"
        ],
        "Penyebab": [
            "Infeksi jamur Phomopsis",
            "Kelembapan tinggi dan kurang sinar matahari"
        ],
        "Saran": [
            "Aplikasi fungisida berbahan tembaga atau mankozeb",
            "Buang daun terinfeksi",
            "Tingkatkan ventilasi udara antar tanaman"
        ]
    }
}

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(page_title="Klasifikasi Penyakit Daun Durian", page_icon="🥭", layout="wide")

# =========================
# Header utama
# =========================
st.markdown("""
    <div style="text-align:center; padding: 2rem 0;">
        <h1 style="color:#1b4332;">🌿 Klasifikasi Penyakit Daun Durian</h1>
        <p style="font-size:18px; color:#2d6a4f;">
            Deteksi penyakit pada daun durian Anda menggunakan teknologi deep learning.<br>
            Upload gambar daun dan dapatkan diagnosis serta rekomendasi penanganan secara instan.
        </p>
    </div>
""", unsafe_allow_html=True)

# =========================
# Petunjuk penggunaan
# =========================
with st.expander("📘 Cara Menggunakan"):
    st.markdown("""
    1. Upload gambar daun durian yang ingin diperiksa  
    2. Klik tombol **Klasifikasi Gambar** untuk memulai analisis  
    3. Lihat hasil diagnosis dan rekomendasi penanganan
    """)

# =========================
# Upload gambar
# =========================
uploaded_file = st.file_uploader("📤 Upload Gambar Daun Durian", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    # Preprocessing gambar
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    if st.button("🔍 Klasifikasi Gambar"):
        with st.spinner("Menganalisis gambar..."):
            preds = model.predict(img_array)
            class_id = np.argmax(preds)
            confidence = np.max(preds)
            label = class_labels[class_id]

        st.success(f"**Hasil Klasifikasi: {label.replace('_', ' ').title()} ({confidence*100:.2f}%)**")

        info = disease_info.get(label, None)
        if info:
            st.subheader("🩺 Deskripsi")
            st.write(info["Deskripsi"])

            st.subheader("🌿 Gejala Umum")
            for g in info["Gejala"]:
                st.markdown(f"- {g}")

            st.subheader("⚠️ Penyebab")
            for p in info["Penyebab"]:
                st.markdown(f"- {p}")

            st.subheader("💡 Saran Penanganan")
            for i, s in enumerate(info["Saran"], start=1):
                st.markdown(f"**{i}.** {s}")
        else:
            st.info("Informasi penyakit tidak tersedia untuk hasil ini.")

else:
    st.info("Silakan upload gambar daun durian terlebih dahulu untuk memulai analisis.")

# =========================
# Footer
# =========================
st.markdown("""
<hr>
<div style="text-align:center; color:gray; font-size:13px;">
    Dibuat dengan ❤️ menggunakan Streamlit | Deteksi Penyakit Daun Durian © 2025
</div>
""", unsafe_allow_html=True)
