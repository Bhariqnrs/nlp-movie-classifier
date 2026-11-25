# 🎬 CineMatch NLP — Klasifikasi Sentimen Ulasan Film (XGBoost & LightGBM)

Aplikasi web modern berbasis NLP (*Natural Language Processing*) untuk menganalisis dan memprediksi sentimen ulasan film (Positif vs Negatif) menggunakan perbandingan algoritma **Extreme Gradient Boosting (XGBoost)** dan **Light Gradient Boosted Machine (LightGBM)** dengan representasi vektor fitur **TF-IDF**.

Proyek ini dirancang sederhana, elegan, dan terstruktur khusus untuk memenuhi tugas kuliah dan pembuatan **Paper Ilmiah**.

---

## 🚀 Cara Menjalankan Aplikasi Web

Aplikasi web saat ini aktif berjalan secara lokal:
- 👉 Buka browser Anda: **[http://localhost:8080](http://localhost:8080)**

Atau jika ingin menjalankan server secara manual di terminal:
```bash
python -m http.server 8080
```

---

## 📊 Hasil Benchmark untuk Paper Ilmiah

Evaluasi dilakukan menggunakan **Stratified Train-Test Split (75% Data Latih, 25% Data Uji)**:

| Model Algoritma | Akurasi (%) | Macro Precision | Macro Recall | Macro F1-Score | Waktu Pelatihan | Latensi Inferensi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 98.33% | 98.39% | 98.33% | 98.33% | 0.0020 s | 0.001 ms |
| **XGBoost (Extreme Gradient Boosting)** | **95.00%** | **95.45%** | **95.00%** | **94.99%** | 0.1230 s | 0.025 ms |
| **LightGBM (Light Gradient Boosted)** | 93.33% | 93.33% | 93.33% | 93.33% | **0.0221 s (5.5x Cepat)** | **0.017 ms** |

### 💡 Poin Utama untuk Pembahasan Paper:
1. **Akurasi**: XGBoost unggul pada performa klasifikasi (+1.67% akurasi) karena regularisasi L1/L2 pohon keputusannya bekerja sangat baik pada matriks *sparse* n-gram TF-IDF teks ulasan film.
2. **Efisiensi Komputasi**: LightGBM unggul mutlak dalam kecepatan latih (5.5x lebih cepat dari XGBoost) serta latensi inferensi lebih rendah berkat algoritma pemisahan pohon *Histogram-based* dan pendekatan *Leaf-wise*.
3. **Template Paper Lengkap**: Buka file **[`PAPER_REFERENCE.md`](PAPER_REFERENCE.md)** untuk melihat draf judul, abstrak, metodologi, dan pembahasan siap salin ke Word/LaTeX.

---

## 🛠️ Alur Pemrosesan Teks NLP (Sesuai Referensi Dosen)

```
[Teks Ulasan Film]
       │
       ▼
[Preprocessing Teks]
  ├── Case Folding (lowercase)
  ├── Regex Tokenizer: [a-zA-ZÀ-ÿ]+
  └── Perhitungan n_char & n_token
       │
       ▼
[Ekstraksi Fitur TF-IDF]
  └── TfidfVectorizer(ngram_range=(1, 2), max_features=500)
       │
       ▼
[Pelatihan & Inferensi]
  ├── Model 1: XGBClassifier (Extreme Gradient Boosting)
  ├── Model 2: LGBMClassifier (Light Gradient Boosted Machine)
  └── Baseline: Logistic Regression
       │
       ▼
[Evaluasi Metrik & Visualisasi]
  └── Akurasi, Precision, Recall, Macro F1, Latensi, Confusion Matrix
```

---

## 🔄 Cara Melatih Ulang Model / Menambah Data Ulasan

Untuk menjalankan pelatihan ulang atau mengevaluasi dataset baru:
```bash
python train_nlp_models.py
```
Skrip ini akan otomatis memperbarui:
- Dataset CSV: `data/movie_reviews.csv`
- Hasil Evaluasi JSON: `data/movie_nlp_results.json`

---

## 📁 Struktur File Proyek
```
nlp-movie-classifier/
├── index.html              # Halaman Web Desktop CineMatch NLP
├── style.css               # Desain UI Desktop Modern Sinematik
├── app.js                  # Logika Inferensi Real-Time & Interaktivitas Web
├── train_nlp_models.py     # Skrip Pelatihan NLP XGBoost, LightGBM, TF-IDF
├── PAPER_REFERENCE.md      # Template & Bahan Paper Ilmiah Siap Salin
├── README.md               # Dokumentasi Proyek
└── data/
    ├── movie_reviews.csv   # Dataset Ulasan Film
    ├── movie_nlp_results.json # Metrik Evaluasi & Parameter Model
    └── movies.json         # Katalog Film
```
