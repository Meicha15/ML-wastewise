# Machine-Learning

# ♻️ WasteWise: Sistem Klasifikasi dan Pengelolaan Sampah Berbasis Deep Learning

**WasteWise** adalah aplikasi berbasis web yang bertujuan untuk mengklasifikasikan **12 kategori sampah**, menentukan apakah sampah tersebut **organik atau anorganik**, serta memberikan **rekomendasi pengelolaan** yang tepat. Sistem ini dibangun menggunakan model **Convolutional Neural Network (CNN)** berbasis **DenseNet** dan diintegrasikan dengan **API Gemini** dari Google untuk menghasilkan saran pengelolaan yang cerdas dan kontekstual.

---

## Tujuan Proyek

1. Membangun model klasifikasi gambar sampah.
2. Mengidentifikasi jenis sampah sebagai **organik** atau **anorganik**.
3. Menampilkan **probabilitas klasifikasi** untuk transparansi prediksi.
4. Memberikan **rekomendasi pengelolaan** berbasis konteks menggunakan **Google Gemini API**.

---
## Jenis Sampah

Model dilatih untuk mengenali **12 jenis sampah berikut**:

| No. | Label Kategori       |
|-----|----------------------|
| 1.  | Alas Kaki            |
| 2.  | Daun                 |
| 3.  | Kaca                 |
| 4.  | Kain/Pakaian         |
| 5.  | Kardus               |
| 6.  | Kayu                 |
| 7.  | Kertas               |
| 8.  | Logam                |
| 9.  | Plastik              |
| 10. | Sampah Elektronik    |
| 11. | Sampah Makanan       |
| 12. | Sterofom             |

## Alur Pengembangan Proyek

### 1. Pengumpulan Data
- Data diperoleh dari:
  - Dataset publik [Kaggle](https://www.kaggle.com/)
  - Scraping manual dari sumber daring

### 2. Pra-pemrosesan
- Resize, normalisasi, augmentasi gambar
- Validasi dan harmonisasi label

### 3. Pembagian Dataset
- **Train**: 70%
- **Validation**: 15%
- **Test**: 15%

📂 Link dataset:
- [Dataset Raw](https://drive.google.com/file/d/1uL27K0c9IbSYzIsztL05NL9Bssd_Qdbe/view?usp=sharing)
- [Dataset Setelah Split](https://drive.google.com/drive/folders/1_IFy7BwcjCnbQNimokk9qzyQfvVGOusp=sharing)

### 4. Pelatihan Model
- Arsitektur: `DenseNet (CNN)`
- Framework: `TensorFlow / Keras`
- Format penyimpanan model:
  - `.keras`
  - `.h5`
  - `.pkl`

### 5. Evaluasi Model
- Akurasi: **95%**
- Evaluasi: confusion matrix

### 6. Integrasi API Gemini
- Menggunakan Gemini API untuk membuat saran pengelolaan berdasarkan hasil klasifikasi.
- Contoh prompt:


---

## Fitur Website

| Fitur | Deskripsi |
|-------|-----------|
| 🖼️ Upload Gambar | Pengguna dapat mengunggah gambar sampah |
| 📌 Prediksi Otomatis | Sistem memprediksi kategori dari 12 label sampah |
| ♻️ Klasifikasi Jenis | Menentukan apakah sampah organik atau anorganik |
| 📊 Probabilitas | Menampilkan tingkat kepercayaan model |
| 💡 Rekomendasi | Memberikan saran pengelolaan menggunakan **Gemini API** |

---

## 🧪 Contoh Output

**Gambar**: Daun kering  
- Kategori: `Daun`  
- Jenis: `Organik`  
- Probabilitas: `91.2%`  
- Saran dari Gemini:
> "Sampah daun kering dapat dijadikan kompos alami atau dimanfaatkan sebagai mulsa pada tanaman."

---
