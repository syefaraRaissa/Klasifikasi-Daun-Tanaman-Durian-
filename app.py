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
MODEL_PATH = "durian1_leaf_disease_model.keras"  # Ganti ke file model baru
CLASS_INDEX_PATH = "class_indices.json"

# (Opsional) URL model di Hugging Face jika nanti kamu mau deploy ke Streamlit Cloud
MODEL_URL = "https://huggingface.co/Syefara/durian-leaf-disease/resolve/main/durian1_leaf_disease_model.keras"

# =========================
# Fungsi download file (jika model di-host online)
# =========================
def download_file_stream(url: str, target_path: str, chunk_size: int = 32768, timeout: int = 120):
    """Download file besar secara streaming"""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = 0
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    return total

# =========================
# Pastikan file class indices tersedia
# =========================
if not os.path.exists(CLASS_INDEX_PATH):
    st.error(f"❌ File `{CLASS_INDEX_PATH}` tidak ditemukan. Pastikan file itu ada di repo kamu.")
    st.stop()

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# =========================
# Download model kalau belum ada (buat Streamlit Cloud)
# =========================
if not os.path.exists(MODEL_PATH):
    try:
        with st.spinner("📦 Mengunduh model dari Hugging Face..."):
            bytes_downloaded = download_file_stream(MODEL_URL, MODEL_PATH)
        st.success(f"✅ Model berhasil diunduh ({bytes_downloaded/1e6:.2f} MB)")
    except Exception as e:
        st.error("❌ Gagal mengunduh model. Periksa URL dan pastikan file bersifat public.")
        st.write("Detail error:", e)
        st.stop()

# =========================
# Load model (.keras) dengan cache
# =========================
@st.cache_resource
def load_keras_model(path):
    try:
        model = tf.keras.models.load_model(path, compile=False)  # Format modern .keras
        return model
    except Exception as e:
        st.error("❌ Gagal memuat model (.keras). Pastikan file valid.")
        st.write("Detail error:", repr(e))
        st.stop()

model = load_keras_model(MODEL_PATH)
st.success("✅ Model berhasil dimuat!")

# =========================
# Informasi penyakit
# =========================
disease_info = {
    "ALGAL_LEAF_SPOT": {
        "Deskripsi": "Penyakit ini disebabkan oleh ganggang hijau parasit (Cephaleuros virescens) yang tumbuh di permukaan daun.",
        "Gejala": ["Bercak hijau keabu-abuan pada permukaan daun", "Permukaan daun terasa kasar", "Pertumbuhan daun terhambat"],
        "Penyebab": ["Kelembapan tinggi dan ventilasi udara buruk", "Kebersihan kebun yang kurang terjaga"],
        "Saran": ["Gunakan fungisida berbahan tembaga secara berkala", "Pangkas daun yang terinfeksi", "Jaga sirkulasi udara antar tanaman"]
    },
    "ALLOCARIDARA_ATTACK": {
        "Deskripsi": "Serangan hama Allocaridara yang menyebabkan daun rusak.",
        "Gejala": ["Bercak kuning tidak beraturan pada daun", "Daun menggulung dan mengering", "Terdapat serangga kecil di bawah permukaan daun"],
        "Penyebab": ["Serangan hama Allocaridara sp.", "Kebersihan kebun yang buruk"],
        "Saran": ["Gunakan insektisida sistemik sesuai dosis", "Monitoring populasi hama rutin", "Pelihara predator alami seperti kepik"]
    },
    "HEALTHY_LEAF": {
        "Deskripsi": "Daun durian sehat tanpa tanda-tanda penyakit atau hama.",
        "Gejala": ["Permukaan daun hijau mengkilap", "Tidak ada bercak atau luka"],
        "Penyebab": ["Tanaman tumbuh optimal dengan nutrisi seimbang"],
        "Saran": ["Pertahankan pola pemupukan seimbang", "Jaga kebersihan kebun", "Pantau kondisi tanaman berkala"]
    },
    "LEAF_BLIGHT": {
        "Deskripsi": "Penyakit bercak daun disebabkan oleh jamur Cercospora atau Alternaria.",
        "Gejala": ["Bercak kecil coklat atau hitam", "Dikelilingi halo kuning", "Daun rontok bila infeksi berat"],
        "Penyebab": ["Infeksi jamur", "Kelembapan tinggi dan percikan air hujan"],
        "Saran": ["Gunakan fungisida sesuai penyebab", "Buang daun terinfeksi", "Perbaiki drainase"]
    },
    "PHOMOPSIS_LEAF_SPOT": {
        "Deskripsi": "Penyakit bercak daun akibat jamur Phomopsis sp.",
        "Gejala": ["Bercak kecil coklat pada daun tua", "Pinggiran bercak lebih gelap", "Daun gugur prematur"],
        "Penyebab": ["Infeksi jamur Phomopsis", "Kelembapan tinggi dan kurang sinar matahari"],
        "Saran": ["Gunakan fungisida tembaga atau mankozeb", "Buang daun terinfeksi", "Tingkatkan ventilasi udara"]
    }
}

# =========================
# UI utama
# =========================
st.set_page_config(page_title="Klasifikasi Penyakit Daun Durian", page_icon="🥭", layout="wide")

st.markdown("""
<div style="text-align:center; padding: 2rem 0;">
    <h1 style="color:#1b4332;">🌿 Klasifikasi Penyakit Daun Durian</h1>
    <p style="font-size:18px; color:#2d6a4f;">
        Upload gambar daun durian Anda untuk mengetahui jenis penyakitnya dan saran penanganannya.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# Upload & Prediksi
# =========================
uploaded_file = st.file_uploader("📤 Upload Gambar Daun Durian", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Gambar yang Diupload", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    if st.button("🔍 Klasifikasi Gambar"):
        with st.spinner("Menganalisis gambar..."):
            preds = model.predict(img_array)
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = class_labels.get(class_id, "UNKNOWN")

        st.success(f"**🌿 Hasil Klasifikasi: {label.replace('_', ' ').title()} ({confidence*100:.2f}%)**")

        info = disease_info.get(label)
        if info:
            st.subheader("🩺 Deskripsi")
            st.write(info["Deskripsi"])

            st.subheader("🌿 Gejala")
            for g in info["Gejala"]:
                st.markdown(f"- {g}")

            st.subheader("⚠️ Penyebab")
            for p in info["Penyebab"]:
                st.markdown(f"- {p}")

            st.subheader("💡 Saran Penanganan")
            for i, s in enumerate(info["Saran"], start=1):
                st.markdown(f"**{i}.** {s}")
        else:
            st.warning("❗ Informasi penyakit tidak ditemukan.")

else:
    st.info("📁 Silakan upload gambar daun durian terlebih dahulu.")

# =========================
# Footer
# =========================
st.markdown("""
<hr>
<div style="text-align:center; color:gray; font-size:13px;">
    Dibuat dengan ❤️ menggunakan Streamlit | Deteksi Penyakit Daun Durian © 2025
</div>
""", unsafe_allow_html=True)
