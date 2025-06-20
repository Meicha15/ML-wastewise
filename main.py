import os
import random
import io  # Diperlukan untuk memproses byte di memori
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image # Pillow library untuk membuka gambar dari memori

# ==============================================================================
# Inisialisasi Aplikasi dan Middleware
# ==============================================================================
app = FastAPI(title="WasteWise ML API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Muat Model dan Data Sekali Saja Saat Startup
# Ini adalah praktik terbaik dan sudah Anda lakukan dengan benar.
# ==============================================================================
print("[INFO] Memuat model machine learning...")
try:
    # Muat model Keras. Ini hanya berjalan sekali per worker.
    model = load_model('save-model/model_dense.keras', compile=False)
    print("[SUCCESS] Model berhasil dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal memuat file model: {e}")
    model = None # Set model ke None jika gagal

print("[INFO] Memuat data saran pengolahan...")
try:
    # Muat dan proses CSV saran. Ini juga hanya berjalan sekali.
    saran_df = pd.read_csv('saran_pengolahan.csv')
    class_names = saran_df['Jenis Sampah'].unique().tolist()
    saran_dict = {row['Jenis Sampah']: [] for _, row in saran_df.iterrows()}
    for _, row in saran_df.iterrows():
        saran_dict[row['Jenis Sampah']].append({
            'metode': row['Metode Pengolahan'],
            'deskripsi': row['Deskripsi']
        })
    print("[SUCCESS] Data saran berhasil dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal memuat file saran_pengolahan.csv: {e}")
    saran_dict = {} # Set ke dictionary kosong jika gagal


golongan_mapping = {
    "Alas Kaki": "Anorganik", "Daun": "Organik", "Kaca": "Anorganik",
    "Kain Pakaian": "Anorganik", "Kardus": "Organik", "Kayu": "Organik",
    "Kertas": "Organik", "Logam": "Anorganik", "Plastik": "Anorganik",
    "Sampah Elektronik": "Anorganik", "Sampah makanan": "Organik", "Sterofoam": "Anorganik"
}

# ==============================================================================
# Fungsi Helper (Tidak ada perubahan signifikan di sini)
# ==============================================================================
def get_saran_dari_dict(jenis_sampah):
    saran_list = saran_dict.get(jenis_sampah, [])
    if saran_list:
        saran_pilihan = random.choice(saran_list)
        return f"{saran_pilihan['metode']}: {saran_pilihan['deskripsi']}"
    else:
        return "Saran pengolahan untuk kategori ini belum tersedia."


# ==============================================================================
# Fungsi Inti Prediksi (TIDAK LAGI MENGGUNAKAN FILE_PATH)
# ==============================================================================
def predict_image_from_bytes(image_bytes: bytes):
    """
    Melakukan prediksi dari data byte gambar yang ada di memori.
    """
    # Buka gambar dari byte stream menggunakan Pillow dan io.BytesIO
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Pastikan format RGB
    # Resize gambar sesuai input model
    img = img.resize((224, 224))
    
    # Konversi gambar ke array numpy
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Lakukan prediksi
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    class_label = class_names[idx]
    confidence = float(np.max(pred)) * 100
    
    golongan = golongan_mapping.get(class_label, "Tidak diketahui")
    saran = get_saran_dari_dict(class_label)
    
    return {
        "Kategori": class_label,
        "Jenis": golongan,
        "Probabilitas": f"{confidence:.1f}%",
        "Saran": saran # Nama field disesuaikan
    }

@app.get("/")
async def read_root():
    """Endpoint untuk memeriksa status API."""
    return {"status": "success", "message": "WasteWise ML API is running"}

@app.post("/api/klasifikasi-sampah")
async def classify_waste_endpoint(file: UploadFile = File(...)):
    """
    Endpoint utama untuk klasifikasi sampah.
    Menerima file gambar dan memprosesnya di memori.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia. Silakan cek log server.")

    # [OPTIMASI] Baca konten file langsung ke memori (bytes)
    image_bytes = await file.read()
    
    # Periksa jika file kosong atau korup
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File gambar tidak boleh kosong.")
        
    try:
        # Panggil fungsi prediksi yang menggunakan byte, bukan path file
        result = predict_image_from_bytes(image_bytes)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        # Tangani error spesifik jika Pillow tidak bisa membuka gambar
        if "cannot identify image file" in str(e):
             raise HTTPException(status_code=400, detail="Format file tidak didukung atau file korup.")
        print(f"[ERROR] Terjadi kesalahan saat prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {e}")

# Bagian ini untuk testing lokal, tidak digunakan oleh PM2 atau uvicorn production
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port)
