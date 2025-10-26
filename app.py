import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import requests

# =========================
# Konfigurasi model & file
# =========================
MODEL_PATH = "durian_leaf_disease_model.h5"
CLASS_INDEX_PATH = "class_indices.json"

# GANTI dengan URL file .h5 di Hugging Face (resolve/main)
# contoh: "https://huggingface.co/username/durian-leaf-disease/resolve/main/durian_leaf_disease_model.h5"
MODEL_URL = "https://huggingface.co/<username>/durian-leaf-disease/resolve/main/durian_leaf_disease_model.h5"

# =========================
# Fungsi util: download dengan chunk (lebih aman)
# =========================
def download_file_stream(url: str, target_path: str, chunk_size: int = 32768, timeout: int = 120):
    """Download file secara streaming dan simpan ke target_path."""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = 0
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
    return total

# =========================
# Pastikan class_indices ada
# =========================
if not os.path.exists(CLASS_INDEX_PATH):
    st.error(f"File `{CLASS_INDEX_PATH}` tidak ditemukan. Pastikan file tersebut ada di repo.")
    st.stop()

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# =========================
# Download model jika perlu
# =========================
if not os.path.exists(MODEL_PATH):
    try:
        with st.spinner("📦 Mengunduh model dari Hugging Face... (ini mungkin memakan waktu beberapa menit untuk file besar)"):
            bytes_downloaded = download_file_stream(MODEL_URL, MODEL_PATH)
        st.success(f"✅ Model terunduh: {bytes_downloaded/1e6:.2f} MB")
    except Exception as e:
        st.error("Gagal mengunduh model. Periksa URL Hugging Face dan pastikan file bersifat public.")
        st.write("Detail error:", repr(e))
        st.stop()

# Tampilkan info ukuran file (debug)
try:
    file_size_mb = os.path.getsize(MODEL_PATH) / 1e6
    st.write(f"Model file size: {file_size_mb:.2f} MB")
except Exception:
    st.write("Tidak dapat membaca ukuran file model.")

# =========================
# Load model dengan cache agar tidak di-load berulang
# =========================
@st.cache_resource
def load_keras_model(path):
    try:
        m = tf.keras.models.load_model(path)
        return m
    except OSError as e:
        # Tampilkan pesan yang membantu
        st.error("Gagal memuat model (.h5). Kemungkinan file korup atau format tidak kompatibel.")
        st.write("Detail OSError:", repr(e))
        st.write("""
            Langkah perbaikan yang disarankan:
            1. Buka file .h5 di browser Hugging Face dan pastikan ukuran sama seperti file lokal asal.  
            2. Download file .h5 ke mesin lokal dan tes `tf.keras.models.load_model('durian_leaf_disease_model.h5')` di komputermu.  
            3. Jika file sangat besar atau load memakan memori, pertimbangkan konversi ke TFLite atau menyimpan sebagai SavedModel (folder).  
        """)
        st.stop()

model = load_keras_model(MODEL_PATH)

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
# UI: Setup page
# =========================
st.set_page_config(page_title="Klasifikasi Penyakit Daun Durian", page_icon="🥭", layout="wide")

st.markdown("""
    <div style="text-align:center; padding: 2rem 0;">
        <h1 style="color:#1b4332;">🌿 Klasifikasi Penyakit Daun Durian</h1>
        <p style="font-size:18px; color:#2d6a4f;">
            Deteksi penyakit pada daun durian Anda menggunakan teknologi deep learning.<br>
            Upload gambar daun dan dapatkan diagnosis serta rekomendasi penanganan secara instan.
        </p>
    </div>
""", unsafe_allow_html=True)

with st.expander("📘 Cara Menggunakan"):
    st.markdown("""
    1. Upload gambar daun durian yang ingin diperiksa  
    2. Klik tombol **Klasifikasi Gambar** untuk memulai analisis  
    3. Lihat hasil diagnosis dan rekomendasi penanganan
    """)

# =========================
# Upload & Prediksi
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
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = class_labels.get(class_id, "UNKNOWN")

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
