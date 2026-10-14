import { useEffect, useState } from "react";
import "./App.css";
import { predictNextCategory, fetchKnownCategories } from "./api";
import CategoryChain from "./components/CategoryChain";
import PredictionCard from "./components/PredictionCard";
import LoadingIndicator from "./components/LoadingIndicator";

const EXAMPLE_INPUT = "Fashion, Shoes, Jacket";

export default function App() {
  const [rawInput, setRawInput] = useState("");
  const [parsedCategories, setParsedCategories] = useState([]);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [knownCategories, setKnownCategories] = useState([]);

  useEffect(() => {
    fetchKnownCategories()
      .then(setKnownCategories)
      .catch(() => {
        // Diam-diam gagal; daftar kategori hanya bersifat informatif (hint UI),
        // tidak menghalangi penggunaan form utama.
      });
  }, []);

  function handleSubmit(event) {
    event.preventDefault();

    const categories = rawInput
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

    if (categories.length === 0) {
      setErrorMessage("Riwayat kategori tidak boleh kosong. Contoh: Fashion, Shoes, Jacket");
      setResult(null);
      return;
    }

    setErrorMessage("");
    setParsedCategories(categories);
    setResult(null);
    setIsLoading(true);

    predictNextCategory(categories, 3)
      .then((data) => {
        setResult(data);
      })
      .catch((error) => {
        const detail =
          error?.response?.data?.detail ||
          "Terjadi kesalahan saat menghubungi server prediksi. Pastikan backend FastAPI sedang berjalan.";
        setErrorMessage(typeof detail === "string" ? detail : "Gagal memproses prediksi.");
      })
      .finally(() => {
        setIsLoading(false);
      });
  }

  function handleUseExample() {
    setRawInput(EXAMPLE_INPUT);
    setErrorMessage("");
  }

  return (
    <div className="page">
      <div className="page__glow" aria-hidden="true" />

      <header className="hero">
        <span className="hero__eyebrow">Model LSTM &middot; Next Category Prediction</span>
        <h1 className="hero__title">
          Tahu langkah belanja
          <br />
          <span className="hero__title-accent">selanjutnya,</span> sebelum mereka tahu.
        </h1>
        <p className="hero__subtitle">
          Masukkan riwayat kategori yang pernah dibeli pengguna, dan model akan menyusuri
          pola tersebut untuk memprediksi tiga kategori paling mungkin dibeli berikutnya.
        </p>
      </header>

      <main className="layout">
        <section className="panel panel--input" aria-label="Form input riwayat kategori">
          <div className="panel__header">
            <h2 className="panel__title">Riwayat Kategori</h2>
            <button type="button" className="ghost-button" onClick={handleUseExample}>
              Pakai contoh
            </button>
          </div>

          <form onSubmit={handleSubmit} className="form">
            <label htmlFor="categories-input" className="form__label">
              Pisahkan setiap kategori dengan koma
            </label>
            <input
              id="categories-input"
              type="text"
              className="form__input"
              placeholder="Fashion, Shoes, Jacket"
              value={rawInput}
              onChange={(event) => {
                setRawInput(event.target.value);
                if (errorMessage) setErrorMessage("");
              }}
              autoComplete="off"
            />

            {errorMessage && (
              <p className="form__error" role="alert">
                {errorMessage}
              </p>
            )}

            <button type="submit" className="primary-button" disabled={isLoading}>
              {isLoading ? "Memprediksi…" : "Predict"}
            </button>
          </form>

          {knownCategories.length > 0 && (
            <div className="known-categories">
              <span className="known-categories__label">Kategori yang dikenali model:</span>
              <div className="known-categories__list">
                {knownCategories.map((cat) => (
                  <span className="known-categories__pill" key={cat}>
                    {cat}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="panel panel--result" aria-label="Hasil prediksi">
          {parsedCategories.length > 0 && (
            <div className="trace">
              <span className="trace__label">Riwayat yang dianalisis</span>
              <CategoryChain categories={parsedCategories} showTarget={!isLoading} />
            </div>
          )}

          {isLoading && <LoadingIndicator />}

          {!isLoading && result && (
            <div className="results">
              <div className="results__header">
                <h2 className="panel__title">Top {result.predictions.length} Prediksi</h2>
                {result.unknown_categories.length > 0 && (
                  <p className="results__note">
                    Kategori tidak dikenali model:{" "}
                    <strong>{result.unknown_categories.join(", ")}</strong>. Prediksi tetap
                    dijalankan menggunakan kategori yang dikenali.
                  </p>
                )}
              </div>

              <div className="results__list">
                {result.predictions.map((item, idx) => (
                  <PredictionCard
                    key={item.category}
                    rank={idx}
                    category={item.category}
                    probability={item.probability}
                    animate
                  />
                ))}
              </div>
            </div>
          )}

          {!isLoading && !result && parsedCategories.length === 0 && (
            <div className="empty-state">
              <p className="empty-state__title">Belum ada prediksi.</p>
              <p className="empty-state__body">
                Masukkan riwayat kategori di sebelah kiri, lalu tekan{" "}
                <strong>Predict</strong> untuk melihat tiga kategori paling mungkin dibeli
                selanjutnya.
              </p>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Didukung oleh model LSTM &middot; FastAPI &middot; React + Vite</span>
      </footer>
    </div>
  );
}
