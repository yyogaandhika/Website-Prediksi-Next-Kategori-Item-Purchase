"""
model_utils.py
----------------
Utility untuk memprediksi kategori produk berikutnya (Next Item Category Prediction).
Menggunakan Google Gemini API (ringan & serverless-ready) dengan fallback dinamis,
tanpa membutuhkan dependensi berat TensorFlow/Keras.
"""

import os
import re
import json
import logging
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger("model_utils")

# 9 Kategori resmi yang dikenali sistem (sesuai dataset e-commerce)
KNOWN_CATEGORIES = [
    "Bag",
    "Dress",
    "Jacket",
    "Outfit",
    "Pants",
    "Shirt",
    "Shoes",
    "Skirt",
    "Socks"
]

# Heuristik asosiasi belanja produk jika API key belum diset atau offline
FALLBACK_ASSOCIATIONS = {
    "shoes": [("Socks", 0.50), ("Pants", 0.30), ("Outfit", 0.15), ("Jacket", 0.05)],
    "jacket": [("Shirt", 0.45), ("Pants", 0.35), ("Shoes", 0.15), ("Bag", 0.05)],
    "shirt": [("Pants", 0.45), ("Jacket", 0.30), ("Shoes", 0.15), ("Outfit", 0.10)],
    "pants": [("Shirt", 0.40), ("Shoes", 0.35), ("Jacket", 0.15), ("Socks", 0.10)],
    "dress": [("Shoes", 0.50), ("Bag", 0.35), ("Jacket", 0.10), ("Outfit", 0.05)],
    "bag": [("Outfit", 0.40), ("Shoes", 0.35), ("Dress", 0.15), ("Jacket", 0.10)],
    "skirt": [("Shirt", 0.45), ("Shoes", 0.35), ("Bag", 0.15), ("Jacket", 0.05)],
    "socks": [("Shoes", 0.60), ("Pants", 0.25), ("Shirt", 0.10), ("Outfit", 0.05)],
    "outfit": [("Shoes", 0.45), ("Bag", 0.30), ("Jacket", 0.20), ("Socks", 0.05)],
}


class NextCategoryPredictor:
    """
    Predictor kategori berikutnya menggunakan Gemini API yang cepat,
    hemat memori, dan siap dideploy ke Vercel Serverless.
    """

    def __init__(self):
        self._loaded = True

    def load(self):
        """Inisialisasi konfigurasi API key."""
        load_dotenv()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()

        if gemini_key:
            logger.info("✅ Next Category Predictor menggunakan Google Gemini API")
        elif groq_key:
            logger.info("✅ Next Category Predictor menggunakan Groq API")
        else:
            logger.warning("⚠️ GEMINI_API_KEY belum diset di .env. Menggunakan fallback heuristic predictor.")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def known_categories(self) -> list[str]:
        """Mengembalikan daftar 9 kategori yang valid."""
        return sorted(KNOWN_CATEGORIES)

    def _match_category(self, cat_name: str) -> str:
        """Mencocokkan nama kategori dengan list resmi secara case-insensitive."""
        target = cat_name.strip().lower()
        for known in KNOWN_CATEGORIES:
            if known.lower() == target:
                return known
        return cat_name.strip().title()

    def _predict_with_gemini(self, recognized_categories: list[str], top_k: int = 3) -> list[dict]:
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan.")

        prompt = (
            "Kamu adalah AI asisten e-commerce untuk analisis kebiasaan belanja (Next Item Category Prediction).\n"
            f"Riwayat pembelian sebelumnya oleh pelanggan: {json.dumps(recognized_categories)}\n"
            f"Daftar kategori barang yang tersedia di katalog toko: {json.dumps(KNOWN_CATEGORIES)}\n\n"
            f"Tugas: Analisis urutan pola belanja tersebut dan tentukan {top_k} kategori teratas yang PALING MUNGKIN dibeli berikutnya dari daftar kategori di atas.\n"
            "Berikan estimasi probabilitas untuk masing-masing rekomendasi (angka desimal antara 0.01 sampai 0.99, dan total jumlahnya mendekati 1.0).\n\n"
            "Format output WAJIB HANYA berupa JSON valid persis seperti ini (tanpa teks pembuka atau markdown tambahan):\n"
            "{\n"
            '  "predictions": [\n'
            '    {"category": "NamaKategori", "probability": 0.60},\n'
            '    {"category": "NamaKategori", "probability": 0.30},\n'
            '    {"category": "NamaKategori", "probability": 0.10}\n'
            "  ]\n"
            "}"
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 300,
                "temperature": 0.2
            }
        }

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        def _is_text_model(name: str) -> bool:
            name_lower = name.lower()
            excluded = ["tts", "audio", "embed", "imagen", "transcription", "realtime", "search"]
            return not any(ex in name_lower for ex in excluded)

        # 1. Otomatis deteksi model teks aktif yang tersedia untuk API key
        target_models = []
        for api_version in ["v1beta", "v1"]:
            try:
                list_url = f"https://generativelanguage.googleapis.com/{api_version}/models"
                res = requests.get(list_url, headers=headers, params={"key": api_key}, timeout=8)
                if res.status_code == 200:
                    data = res.json().get("models", [])
                    flash_models = [
                        (api_version, m["name"].replace("models/", ""))
                        for m in data
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                        and _is_text_model(m.get("name", ""))
                        and "flash" in m.get("name", "").lower()
                    ]
                    other_models = [
                        (api_version, m["name"].replace("models/", ""))
                        for m in data
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                        and _is_text_model(m.get("name", ""))
                        and "flash" not in m.get("name", "").lower()
                    ]
                    target_models.extend(flash_models + other_models)
                    if target_models:
                        break
            except Exception:
                pass

        # 2. Fallback daftar nama model jika ListModels tidak merespons
        if not target_models:
            fallback_names = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro",
                "gemini-1.5-pro-latest"
            ]
            target_models = [("v1beta", name) for name in fallback_names] + [("v1", name) for name in fallback_names]

        last_error = ""
        for api_version, model_name in target_models:
            url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    params={"key": api_key},
                    json=payload,
                    timeout=20
                )
                if response.status_code == 200:
                    res_json = response.json()
                    raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return self._parse_predictions_json(raw_text, top_k)
                elif response.status_code == 403:
                    raise Exception(f"Gemini API Error (403): {response.text}")
                else:
                    last_error = response.text
            except Exception as e:
                if "403" in str(e):
                    raise e
                last_error = str(e)

        raise Exception(f"Semua model Gemini gagal merespons: {last_error}")

    def _predict_with_groq(self, recognized_categories: list[str], top_k: int = 3) -> list[dict]:
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY tidak ditemukan.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        prompt = (
            f"Pelanggan membeli riwayat kategori: {json.dumps(recognized_categories)}.\n"
            f"Kategori yang tersedia: {json.dumps(KNOWN_CATEGORIES)}.\n"
            f"Berikan {top_k} prediksi kategori teratas berikutnya dengan estimasi probabilitas (0.01 - 0.99).\n"
            'Kembalikan HANYA format JSON: {"predictions": [{"category": "...", "probability": 0.5}, ...]}'
        )

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a Next Item Category Prediction assistant. Always return pure JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return self._parse_predictions_json(content, top_k)
        raise Exception(f"Groq API Error: {response.text}")

    def _parse_predictions_json(self, raw_text: str, top_k: int) -> list[dict]:
        """Membersihkan markdown dan mengekstrak array predictions dari JSON output."""
        cleaned = raw_text
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1)

        data = json.loads(cleaned.strip())
        items = data.get("predictions", [])
        if not items and isinstance(data, list):
            items = data

        results = []
        seen = set()
        for it in items:
            cat_name = self._match_category(str(it.get("category", "")))
            if not cat_name or cat_name in seen:
                continue
            seen.add(cat_name)
            prob = float(it.get("probability", 0.0))
            if prob > 1.0:
                prob = prob / 100.0
            results.append({
                "category": cat_name,
                "probability": round(min(max(prob, 0.01), 0.99), 4)
            })

        # Urutkan berdasarkan probabilitas tertinggi
        results.sort(key=lambda x: x["probability"], reverse=True)

        # Jika kurang dari top_k, lengkapi dengan kategori lain
        if len(results) < top_k:
            for cat in KNOWN_CATEGORIES:
                if cat not in seen:
                    results.append({"category": cat, "probability": 0.05})
                    seen.add(cat)
                    if len(results) >= top_k:
                        break

        results = results[:top_k]

        # Normalisasi probabilitas agar jumlahnya mendekati 1.0
        total_p = sum(r["probability"] for r in results)
        if total_p > 0:
            for r in results:
                r["probability"] = round(r["probability"] / total_p, 4)

        return results

    def _predict_fallback(self, recognized_categories: list[str], top_k: int = 3) -> list[dict]:
        """
        Prediksi fallback cerdas berbasis asosiasi kebiasaan belanja retail
        jika API key belum diset atau koneksi internet bermasalah.
        """
        last_cat = recognized_categories[-1].lower() if recognized_categories else "shirt"
        candidates = FALLBACK_ASSOCIATIONS.get(last_cat, [])

        results = []
        used = set()
        for cat, prob in candidates:
            matched = self._match_category(cat)
            results.append({"category": matched, "probability": prob})
            used.add(matched)
            if len(results) >= top_k:
                break

        # Jika masih kurang dari top_k, tambahkan kategori yang belum ada
        if len(results) < top_k:
            for cat in KNOWN_CATEGORIES:
                if cat not in used:
                    results.append({"category": cat, "probability": 0.05})
                    used.add(cat)
                    if len(results) >= top_k:
                        break

        results = results[:top_k]
        total_p = sum(r["probability"] for r in results)
        if total_p > 0:
            for r in results:
                r["probability"] = round(r["probability"] / total_p, 4)

        return results

    def predict_top_k(self, categories: list[str], top_k: int = 3) -> dict:
        """
        Menjalankan prediksi kategori produk berikutnya.
        """
        known_map = {k.lower(): k for k in KNOWN_CATEGORIES}

        recognized = []
        unknown = []

        for cat in categories:
            cleaned = cat.strip()
            if not cleaned:
                continue
            if cleaned.lower() in known_map:
                recognized.append(known_map[cleaned.lower()])
            else:
                unknown.append(cleaned)

        if not recognized:
            return {
                "predictions": [],
                "recognized_categories": [],
                "unknown_categories": unknown,
            }

        # Coba prediksi dengan Gemini API
        if os.getenv("GEMINI_API_KEY"):
            try:
                preds = self._predict_with_gemini(recognized, top_k=top_k)
                return {
                    "predictions": preds,
                    "recognized_categories": recognized,
                    "unknown_categories": unknown,
                }
            except Exception as e:
                logger.warning("Gagal dengan Gemini API: %s. Mencoba fallback.", e)

        # Coba dengan Groq jika ada
        if os.getenv("GROQ_API_KEY"):
            try:
                preds = self._predict_with_groq(recognized, top_k=top_k)
                return {
                    "predictions": preds,
                    "recognized_categories": recognized,
                    "unknown_categories": unknown,
                }
            except Exception as e:
                logger.warning("Gagal dengan Groq API: %s. Mencoba fallback.", e)

        # Fallback lokal cerdas (tanpa TensorFlow)
        logger.info("Menggunakan fallback asosiasi pola belanja lokal.")
        preds = self._predict_fallback(recognized, top_k=top_k)
        return {
            "predictions": preds,
            "recognized_categories": recognized,
            "unknown_categories": unknown,
        }


# Singleton predictor
predictor = NextCategoryPredictor()
