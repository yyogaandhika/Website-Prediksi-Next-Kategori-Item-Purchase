"""
model_utils.py
----------------
Utility untuk memuat model LSTM (.h5) dan tokenizer (.pkl) yang dilatih
menggunakan TensorFlow/Keras versi lama (format Sequential + Keras
preprocessing.text.Tokenizer), serta melakukan preprocessing & prediksi
yang konsisten dengan proses training.

Catatan kompatibilitas:
Model dan tokenizer ini dibuat dengan API `keras.preprocessing.text.Tokenizer`
dan layer LSTM versi lama (mengandung argumen `time_major` yang sudah
dihapus di Keras 3). Agar tetap bisa dimuat tanpa perlu re-training,
modul ini menggunakan paket `tf-keras` (implementasi Keras 2 legacy yang
masih disediakan resmi oleh tim TensorFlow) dan membuat "shim" module
bernama `keras.preprocessing.text` yang menunjuk ke kelas Tokenizer pada
`tf_keras`, supaya proses unpickling tokenizer.pkl berhasil.
"""

import os
import sys
import types
import pickle
import logging

# WAJIB di-set sebelum import tensorflow/tf_keras, agar TensorFlow
# menggunakan implementasi Keras 2 (legacy) yang kompatibel dengan
# format model lama (Sequential h5 dengan LSTM time_major).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # kurangi log TensorFlow

import numpy as np

logger = logging.getLogger("model_utils")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "next_category_lstm.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Harus sama dengan input_length saat training (didapat dari arsitektur model).
MAX_SEQUENCE_LENGTH = 34


def _install_keras_tokenizer_shim():
    """
    tokenizer.pkl dipickle menggunakan kelas
    `keras.preprocessing.text.Tokenizer`. Modul tersebut sudah tidak ada
    di Keras 3, sehingga unpickle langsung akan gagal dengan
    ModuleNotFoundError. Fungsi ini mendaftarkan modul palsu (shim) ke
    sys.modules yang menunjuk ke implementasi Tokenizer dari `tf_keras`
    (Keras 2 legacy), yang strukturnya identik dengan versi lama.
    """
    if "keras.preprocessing.text" in sys.modules:
        return  # shim sudah terpasang

    import tf_keras.preprocessing.text as legacy_text

    keras_mod = sys.modules.get("keras") or types.ModuleType("keras")
    keras_preprocessing_mod = types.ModuleType("keras.preprocessing")
    keras_preprocessing_text_mod = types.ModuleType("keras.preprocessing.text")

    keras_preprocessing_text_mod.Tokenizer = legacy_text.Tokenizer
    keras_preprocessing_mod.text = keras_preprocessing_text_mod
    keras_mod.preprocessing = keras_preprocessing_mod

    sys.modules["keras"] = keras_mod
    sys.modules["keras.preprocessing"] = keras_preprocessing_mod
    sys.modules["keras.preprocessing.text"] = keras_preprocessing_text_mod


class NextCategoryPredictor:
    """
    Wrapper yang membungkus model LSTM + tokenizer untuk melakukan
    prediksi kategori produk berikutnya berdasarkan riwayat kategori.
    """

    def __init__(self, model_path: str = MODEL_PATH, tokenizer_path: str = TOKENIZER_PATH):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.index_word = {}
        self._loaded = False

    def load(self):
        """Memuat model .h5 dan tokenizer .pkl ke memori. Dipanggil sekali saat startup."""
        _install_keras_tokenizer_shim()

        import tf_keras  # import setelah shim & env var terpasang

        logger.info("Memuat tokenizer dari %s", self.tokenizer_path)
        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        # index_word: mapping index -> nama kategori (huruf kecil, sesuai training)
        self.index_word = dict(self.tokenizer.index_word)

        logger.info("Memuat model dari %s", self.model_path)
        self.model = tf_keras.models.load_model(self.model_path)

        self._loaded = True
        logger.info(
            "Model & tokenizer berhasil dimuat. Vocab size=%d, max_len=%d",
            len(self.index_word),
            MAX_SEQUENCE_LENGTH,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def known_categories(self):
        """Daftar kategori yang dikenali oleh tokenizer (sesuai data training)."""
        # index_word menyimpan key dalam huruf kecil; kembalikan dengan title case
        return sorted({v.title() for v in self.index_word.values()})

    def _preprocess(self, categories):
        """
        Melakukan preprocessing yang identik dengan proses training:
        1. Gabungkan daftar kategori menjadi satu string dipisah spasi
           (karena Tokenizer Keras bekerja pada level kata yang dipisah
           spasi setelah proses filtering & lowercasing internal).
        2. texts_to_sequences -> ubah token menjadi urutan index integer.
        3. pad_sequences -> padding 'pre' hingga panjang MAX_SEQUENCE_LENGTH,
           sesuai input_length yang digunakan saat training model.
        """
        from tf_keras.preprocessing.sequence import pad_sequences

        text = " ".join(categories)
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="pre")
        return padded, sequence[0]

    def predict_top_k(self, categories, top_k: int = 3):
        """
        Menjalankan prediksi kategori berikutnya.

        Args:
            categories: list[str] riwayat kategori, contoh ["Fashion", "Shoes", "Jacket"]
            top_k: jumlah kandidat teratas yang dikembalikan

        Returns:
            dict berisi:
                - predictions: list of {category, probability}
                - unknown_categories: kategori input yang tidak dikenali tokenizer (OOV)
                - recognized_categories: kategori input yang dikenali tokenizer
        """
        if not self._loaded:
            raise RuntimeError("Model belum dimuat. Panggil load() terlebih dahulu.")

        padded, raw_sequence = self._preprocess(categories)

        # Deteksi kategori yang tidak dikenal tokenizer (OOV / hasil filter jadi kosong)
        # Tokenizer Keras tanpa oov_token akan membuang token yang tidak ada di word_index,
        # sehingga panjang raw_sequence bisa lebih kecil dari jumlah kategori input.
        recognized = []
        unknown = []
        word_index = self.tokenizer.word_index
        for cat in categories:
            cleaned = cat.strip().lower()
            if cleaned in word_index:
                recognized.append(cat.strip())
            else:
                unknown.append(cat.strip())

        probabilities = self.model.predict(padded, verbose=0)[0]

        # index 0 pada output biasanya merepresentasikan padding/tidak ada kategori,
        # sehingga dikecualikan dari kandidat top-k yang ditampilkan ke pengguna.
        candidate_indices = [i for i in range(len(probabilities)) if i != 0]
        candidate_indices.sort(key=lambda i: probabilities[i], reverse=True)
        top_indices = candidate_indices[:top_k]

        predictions = []
        for idx in top_indices:
            category_name = self.index_word.get(idx, f"unknown_{idx}")
            predictions.append(
                {
                    "category": category_name.title(),
                    "probability": float(probabilities[idx]),
                }
            )

        return {
            "predictions": predictions,
            "recognized_categories": recognized,
            "unknown_categories": unknown,
        }


# Instance singleton yang akan dipakai oleh main.py
predictor = NextCategoryPredictor()
