# Next Category Predictor

Website prediksi kategori produk berikutnya untuk e-commerce berbasis AI (Google Gemini API dengan fallback otomatis), hemat resource, dan siap dideploy langsung ke **Vercel Serverless**.

- **Backend**: FastAPI (Python, Serverless-ready)
- **Engine AI**: Google Gemini API (tanpa dependensi berat TensorFlow)
- **Frontend**: React + Vite + Axios
- **Deployment**: Vercel (Frontend & Backend dalam satu repository)

---

## Struktur Project

```
.
├── api/
│   ├── index.py           # Vercel Serverless Function entrypoint
│   └── requirements.txt   # Dependensi untuk Vercel
├── backend/
│   ├── main.py            # FastAPI app: /predict, /categories, /health
│   ├── model_utils.py     # Gemini AI & Dynamic Fallback Predictor
│   └── requirements.txt
├── src/
│   ├── components/
│   │   ├── CategoryChain.jsx     # Visualisasi rantai riwayat kategori
│   │   ├── PredictionCard.jsx    # Kartu hasil prediksi + progress bar
│   │   └── LoadingIndicator.jsx
│   ├── api.js             # Konfigurasi Axios ke backend
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── vercel.json            # Konfigurasi deployment Vercel
├── requirements.txt       # Root requirements untuk Vercel build
├── vite.config.js         # Vite config dengan proxy /api
├── package.json
└── .env.example
```

---

## 9 Kategori Produk Resmi

Sistem mengenali 9 kategori belanja e-commerce:
**Bag, Dress, Jacket, Outfit, Pants, Shirt, Shoes, Skirt, Socks**

Jika user memasukkan kategori di luar daftar tersebut (misal: "Fashion"), sistem tetap memprosesnya namun menandainya di `unknown_categories` pada response.

---

## 1. Menjalankan Backend (Lokal)

1. Masuk ke folder backend dan install dependensi ringan (tanpa TensorFlow):
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Buat file `.env` di root project atau di folder backend:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   > Dapatkan API key gratis di: [Google AI Studio](https://aistudio.google.com/app/apikey)

3. Jalankan server FastAPI:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Backend aktif di: **http://localhost:8000** (Swagger UI di **/docs**)

---

## 2. Menjalankan Frontend (Lokal)

Di terminal terpisah:
```bash
npm install
npm run dev
```
Buka **http://localhost:5173** di browser. Panggilan API `/api` otomatis diteruskan ke backend FastAPI port 8000.

---

## 3. Deploy ke Vercel (1-Click)

Project ini sudah dilengkapi konfigurasi serverless:
1. Push repository ke GitHub.
2. Buka [Vercel Dashboard](https://vercel.com/) -> **Add New Project** -> Pilih repo ini.
3. Di bagian **Environment Variables**, tambahkan:
   - `GEMINI_API_KEY`: *(isi dengan API Key Gemini kamu)*
4. Klik **Deploy**.
   Vercel akan otomatis mem-build frontend Vite dan menjalankan backend FastAPI sebagai serverless function!
