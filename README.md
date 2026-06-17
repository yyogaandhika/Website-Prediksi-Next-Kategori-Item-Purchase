# Next Category Predictor

Website prediksi kategori produk berikutnya untuk e-commerce, menggunakan model **LSTM** yang sudah dilatih sebelumnya (`next_category_lstm.h5` + `tokenizer.pkl`).

Backend: **FastAPI** (Python) · Frontend: **React + Vite** · Komunikasi: **Axios**

---

## Struktur Project

```
.
├── backend/
│   ├── main.py            # FastAPI app: endpoint /predict, /categories, /health
│   ├── model_utils.py      # Loader model & tokenizer + preprocessing + prediksi
│   ├── requirements.txt
│   └── model/
│       ├── next_category_lstm.h5
│       └── tokenizer.pkl
├── src/
│   ├── components/
│   │   ├── CategoryChain.jsx     # Visualisasi rantai riwayat kategori
│   │   ├── PredictionCard.jsx    # Kartu hasil prediksi + progress bar
│   │   └── LoadingIndicator.jsx
│   ├── api.js               # Konfigurasi Axios ke backend
│   ├── App.jsx
│   ├── App.css
│   ├── index.css
│   └── main.jsx
├── public/
│   └── favicon.svg
├── index.html
├── package.json
├── vite.config.js
└── .env.example
```

---

## Catatan penting tentang model

Model dan tokenizer yang diunggah dibuat dengan `keras.preprocessing.text.Tokenizer` dan layer LSTM dari versi Keras lama (mengandung argumen `time_major` yang sudah dihapus di Keras 3). Agar file `.h5` dan `.pkl` ini tetap bisa dimuat **tanpa perlu re-training**, backend menggunakan paket `tf-keras` (implementasi Keras 2 resmi dari tim TensorFlow) dan sebuah *compatibility shim* di `model_utils.py` agar proses `pickle.load()` pada tokenizer berhasil. Ini sudah diuji dan berjalan normal — tidak perlu langkah tambahan dari sisi kamu, cukup install `requirements.txt`.

Vocabulary tokenizer hasil training berisi 9 kategori: **Outfit, Socks, Shirt, Dress, Pants, Bag, Jacket, Shoes, Skirt**. Jika user memasukkan kategori yang tidak ada dalam daftar ini (misalnya "Fashion"), backend akan tetap memprosesnya tapi menandainya sebagai `unknown_categories` di response — prediksi hanya dihitung dari kategori yang dikenali. Frontend menampilkan daftar kategori valid ini secara otomatis (diambil dari endpoint `/categories`) sebagai panduan untuk user.

---

## 1. Menjalankan Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Backend akan berjalan di **http://localhost:8000**.

Cek dokumentasi interaktif (Swagger UI) di **http://localhost:8000/docs**.

### Endpoint yang tersedia

| Method | Endpoint      | Deskripsi                                            |
|--------|---------------|-------------------------------------------------------|
| GET    | `/`           | Info dasar API                                        |
| GET    | `/health`     | Status server & status model                          |
| GET    | `/categories` | Daftar kategori yang dikenali model                    |
| POST   | `/predict`    | Prediksi top-K kategori berikutnya                      |

Contoh request `/predict`:

```json
{
  "categories": ["Fashion", "Shoes", "Jacket"],
  "top_k": 3
}
```

Contoh response:

```json
{
  "input_categories": ["Fashion", "Shoes", "Jacket"],
  "recognized_categories": ["Shoes", "Jacket"],
  "unknown_categories": ["Fashion"],
  "predictions": [
    { "category": "Shoes", "probability": 0.6229952573776245 },
    { "category": "Jacket", "probability": 0.3300647437572479 },
    { "category": "Outfit", "probability": 0.03525746986269951 }
  ]
}
```

---

## 2. Menjalankan Frontend (React + Vite)

Buka terminal baru (biarkan backend tetap berjalan):

```bash
npm install
npm run dev
```

Frontend akan berjalan di **http://localhost:5173**.

Jika backend dijalankan di host/port berbeda, salin `.env.example` menjadi `.env` dan sesuaikan:

```
VITE_API_BASE_URL=http://localhost:8000
```

---

## 3. Cara Pakai

1. Buka **http://localhost:5173** di browser.
2. Masukkan riwayat kategori, dipisahkan koma, contoh: `Fashion, Shoes, Jacket`.
3. Klik **Predict**.
4. Tunggu loading indicator selesai — hasil top-3 kategori berikutnya akan muncul dalam bentuk kartu dengan persentase probabilitas dan progress bar.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'keras.preprocessing.text'`** — pastikan `tf-keras` ikut terinstal (`pip install -r requirements.txt`); jangan instal ulang versi `tensorflow`/`keras` secara manual yang berbeda dari `requirements.txt`.
- **CORS error di browser** — pastikan frontend diakses dari `http://localhost:5173` (origin ini sudah didaftarkan di `main.py`). Jika port berbeda, tambahkan ke `allow_origins` pada `backend/main.py`.
- **Prediksi gagal / "Tidak ada kategori yang dikenali model"** — pastikan minimal satu kategori yang diketik cocok dengan daftar di endpoint `/categories` (huruf besar/kecil tidak masalah, backend otomatis lowercase).
