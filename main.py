import random
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil
import pandas as pd  # Import pandas untuk baca CSV

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # izinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

model = load_model('save-model/model_dense.keras', compile=False)

class_names = ['Alas Kaki', 'Daun', 'Kaca', 'Kain Pakaian', 'Kardus', 'Kayu',
               'Kertas', 'Logam', 'Plastik', 'Sampah Elektronik', 'Sampah makanan', 'Sterofoam']

golongan_mapping = {
    "Alas Kaki": "Anorganik",
    "Daun": "Organik",
    "Kaca": "Anorganik",
    "Kain Pakaian": "Anorganik",
    "Kardus": "Organik",
    "Kayu": "Organik",
    "Kertas": "Organik",
    "Logam": "Anorganik",
    "Plastik": "Anorganik",
    "Sampah Elektronik": "Anorganik",
    "Sampah makanan": "Organik",
    "Sterofoam": "Anorganik"
}

# Load CSV saran pengolahan (pastikan path benar)
saran_df = pd.read_csv('saran_pengolahan.csv')
saran_dict = {}

# Menyimpan semua saran dalam bentuk list berdasarkan jenis sampah
for _, row in saran_df.iterrows():
    if row['Jenis Sampah'] not in saran_dict:
        saran_dict[row['Jenis Sampah']] = []
    saran_dict[row['Jenis Sampah']].append({
        'metode': row['Metode Pengolahan'],
        'deskripsi': row['Deskripsi']
    })

# Fungsi untuk memilih saran secara acak
def get_saran_dari_csv(jenis_sampah):
    saran_list = saran_dict.get(jenis_sampah, [])
    if saran_list:
        saran_pilihan = random.choice(saran_list)  # Memilih saran secara acak
        return {
            'metode': saran_pilihan['metode'],
            'deskripsi': saran_pilihan['deskripsi']
        }
    else:
        return {
            'metode': 'Metode pengolahan belum tersedia',
            'deskripsi': 'Deskripsi pengolahan belum tersedia.'
        }

def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    class_label = class_names[idx]
    confidence = float(np.max(pred)) * 100
    golongan = golongan_mapping.get(class_label, "Tidak diketahui")
    saran = get_saran_dari_csv(class_label)

    return {
        "Jenis Sampah": class_label,
        "Kategori": golongan,
        "Probabilitas": f"{confidence:.1f}%",
        "Metode Pengolahan": saran['metode'],
        "Deskripsi Pengolahan": saran['deskripsi']
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/klasifikasi-sampah")
async def classify_waste(request: Request, file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = predict_image(temp_file)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_file)

    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.post("/api/klasifikasi-sampah")
async def classify_waste_json(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = predict_image(temp_file)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_file)

    return JSONResponse(content=result, status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
