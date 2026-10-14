"""
main.py
--------
FastAPI backend untuk Next Item Category Prediction.
Mendukung eksekusi lokal dan deployment serverless ke Vercel.

Endpoint:
- GET  / dan /api/               -> info dasar API
- GET  /health dan /api/health   -> health check (status predictor)
- GET  /categories dan /api/categories -> daftar 9 kategori resmi
- POST /predict dan /api/predict       -> prediksi top-K kategori berikutnya
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from model_utils import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: inisialisasi predictor
    logger.info("Menginisialisasi Next Category Predictor...")
    predictor.load()
    logger.info("Predictor siap digunakan.")
    yield


app = FastAPI(
    title="Next Category Prediction API",
    description="API untuk memprediksi kategori produk berikutnya menggunakan AI.",
    version="1.0.0",
    lifespan=lifespan,
)

# Konfigurasi CORS agar frontend React (Vite / Vercel) dapat mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    probability: float  # nilai 0..1


class PredictResponse(BaseModel):
    input_categories: list[str]
    recognized_categories: list[str]
    unknown_categories: list[str]
    predictions: list[PredictionItem]


# Buat Router untuk mendukung endpoint baik dengan atau tanpa prefix /api
router = APIRouter()


@router.get("/")
def root():
    return {
        "status": "online",
        "message": "Next Category Prediction API aktif.",
        "docs": "/docs",
        "endpoints": ["/health", "/categories", "/predict"],
    }


@router.get("/health")
def health_check():
    return {
        "status": "ok" if predictor.is_loaded else "model_not_loaded",
        "model_loaded": predictor.is_loaded,
    }


@router.get("/categories")
def get_known_categories():
    """Mengembalikan daftar kategori yang dikenali oleh sistem."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Predictor belum siap.")
    return {"categories": predictor.known_categories()}


@router.post("/predict", response_model=PredictResponse)
def predict_next_category(payload: PredictRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Predictor belum siap, coba lagi sebentar.")

    try:
        result = predictor.predict_top_k(payload.categories, top_k=payload.top_k)
    except Exception as exc:
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


# Pasang router ke root dan ke prefix /api
app.include_router(router)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
