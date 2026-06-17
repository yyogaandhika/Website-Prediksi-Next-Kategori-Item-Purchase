"""
main.py
--------
FastAPI backend untuk Next Item Category Prediction.

Endpoint:
- GET  /                -> info dasar API
- GET  /health           -> health check (status model)
- GET  /categories        -> daftar kategori yang dikenali model
- POST /predict           -> prediksi top-K kategori berikutnya

Jalankan dengan:
    uvicorn main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from model_utils import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: muat model & tokenizer sekali saja, simpan di memori.
    logger.info("Memuat model LSTM dan tokenizer...")
    predictor.load()
    logger.info("Model siap digunakan.")
    yield
    # Shutdown: tidak ada resource khusus yang perlu dibersihkan.


app = FastAPI(
    title="Next Category Prediction API",
    description="API untuk memprediksi kategori produk berikutnya menggunakan model LSTM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Konfigurasi CORS agar frontend React (Vite, default port 5173) dapat
# mengakses API ini. Untuk produksi, ganti allow_origins dengan domain
# frontend yang sebenarnya.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    categories: list[str] = Field(
        ...,
        description="Riwayat kategori pembelian, contoh: ['Fashion', 'Shoes', 'Jacket']",
        min_length=1,
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Jumlah kandidat prediksi teratas")

    @field_validator("categories")
    @classmethod
    def categories_not_empty(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if v and v.strip()]
        if not cleaned:
            raise ValueError("Daftar kategori tidak boleh kosong.")
        return cleaned


class PredictionItem(BaseModel):
    category: str
    probability: float  # nilai 0..1, frontend yang mengonversi ke persen


class PredictResponse(BaseModel):
    input_categories: list[str]
    recognized_categories: list[str]
    unknown_categories: list[str]
    predictions: list[PredictionItem]


@app.get("/")
def root():
    return {
        "message": "Next Category Prediction API aktif.",
        "docs": "/docs",
        "endpoints": ["/health", "/categories", "/predict"],
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok" if predictor.is_loaded else "model_not_loaded",
        "model_loaded": predictor.is_loaded,
    }


@app.get("/categories")
def get_known_categories():
    """Mengembalikan daftar kategori yang dikenali oleh model (sesuai data training)."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    return {"categories": predictor.known_categories()}


@app.post("/predict", response_model=PredictResponse)
def predict_next_category(payload: PredictRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum siap, coba lagi sebentar.")

    try:
        result = predictor.predict_top_k(payload.categories, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gagal melakukan prediksi")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {exc}") from exc

    if not result["recognized_categories"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Tidak ada kategori yang dikenali model. "
                "Gunakan salah satu dari daftar kategori yang valid (lihat endpoint /categories)."
            ),
        )

    return PredictResponse(
        input_categories=payload.categories,
        recognized_categories=result["recognized_categories"],
        unknown_categories=result["unknown_categories"],
        predictions=result["predictions"],
    )
