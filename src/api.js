import axios from "axios";

// Base URL backend FastAPI. Saat development, default-nya localhost:8000.
// Untuk deployment, ganti via environment variable VITE_API_BASE_URL.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Mengirim riwayat kategori ke backend dan menerima top-K prediksi kategori berikutnya.
 * @param {string[]} categories - daftar kategori riwayat, contoh: ["Fashion", "Shoes", "Jacket"]
 * @param {number} topK - jumlah kandidat prediksi teratas (default 3)
 */
export async function predictNextCategory(categories, topK = 3) {
  const response = await apiClient.post("/predict", {
    categories,
    top_k: topK,
  });
  return response.data;
}

export async function fetchKnownCategories() {
  const response = await apiClient.get("/categories");
  return response.data.categories;
}

export async function checkHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}
