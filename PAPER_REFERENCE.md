# 📄 Panduan & Bahan Paper Ilmiah NLP (Klasifikasi Film)

**Judul Paper Usulan:**  
> **Komparasi Kinerja Algoritma Extreme Gradient Boosting (XGBoost) dan Light Gradient Boosting Machine (LightGBM) Berbasis TF-IDF untuk Analisis Sentimen Ulasan Film**

---

## 1. Abstrak
Analisis sentimen ulasan film merupakan salah satu cabang *Natural Language Processing* (NLP) yang berperan penting dalam memahami persepsi dan respon penonton terhadap karya sinematik. Penelitian ini membandingkan kinerja dua algoritma *gradient boosting* terkemuka, yaitu **Extreme Gradient Boosting (XGBoost)** dan **Light Gradient Boosted Machine (LightGBM)**, yang dikombinasikan dengan ekstraksi fitur *Term Frequency-Inverse Document Frequency* (TF-IDF) unigram dan bigram (`ngram_range=(1,2)`). Evaluasi dilakukan dengan validasi bertingkat (*stratified train-test split* 75:25) dan dibandingkan dengan model dasar (*baseline*) Regresi Logistik. Hasil eksperimen membuktikan bahwa XGBoost mencapai akurasi sebesar **95.00%** dan Macro F1-Score **94.99%**, sedangkan LightGBM mencapai akurasi **93.33%** dan Macro F1-Score **93.33%**. Meskipun XGBoost unggul tipis dalam akurasi (+1.67%), LightGBM memiliki efisiensi komputasi pelatihan yang jauh lebih cepat (0.022 detik vs 0.123 detik atau **5.5x lebih cepat**) serta latensi inferensi yang lebih rendah (0.017 ms vs 0.025 ms per sampel).

**Kata Kunci:** *Analisis Sentimen, NLP, Ulasan Film, TF-IDF, XGBoost, LightGBM, Text Classification.*

---

## 2. Metodologi Penelitian
1. **Pengumpulan & Penyusunan Dataset**: Dataset ulasan film mencakup spektrum genre (Action, Sci-Fi, Drama, Horor, Komedi, Thriller, Romance, dll) dengan label biner: Positif (1) dan Negatif (0).
2. **Preprocessing Teks**:
   - Case folding (mengubah teks ke huruf kecil).
   - Regex Tokenization: `re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())`.
   - Analisis panjang karakter (`n_char`) dan jumlah token (`n_token`).
3. **Ekstraksi Fitur**:
   - TF-IDF Vectorizer dengan rentang n-gram (1, 2) untuk menangkap konteks frasa seperti "sangat memukau", "plot twist", "sangat membosankan", "tidak seram".
   - `max_features = 500`.
4. **Pembagian Data**:
   - Stratified Train-Test Split (75% Train, 25% Test).
5. **Pelatihan Algoritma**:
   - **XGBoost Classifier**: `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`.
   - **LightGBM Classifier**: `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`, `num_leaves=15`, `subsample=0.8`.
   - **Baseline**: Logistic Regression (`max_iter=1000`).

---

## 3. Tabel Hasil Evaluasi untuk Paper

| Model Evaluasi | Akurasi (%) | Precision (Macro %) | Recall (Macro %) | F1-Score (Macro %) | Waktu Latih (detik) | Latensi Inferensi (ms/item) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 98.33% | 98.39% | 98.33% | 98.33% | 0.0020 s | 0.001 ms |
| **XGBoost** | **95.00%** | **95.45%** | **95.00%** | **94.99%** | 0.1230 s | 0.025 ms |
| **LightGBM** | 93.33% | 93.33% | 93.33% | 93.33% | **0.0221 s (5.5x Cepat)** | **0.017 ms** |

---

## 4. Confusion Matrix

### XGBoost (Akurasi 95.00%):
- **True Negative (TN)**: 30 ulasan negatif diprediksi benar.
- **False Positive (FP)**: 0 ulasan negatif salah diprediksi positif.
- **False Negative (FN)**: 3 ulasan positif salah diprediksi negatif.
- **True Positive (TP)**: 27 ulasan positif diprediksi benar.

### LightGBM (Akurasi 93.33%):
- **True Negative (TN)**: 27 ulasan negatif diprediksi benar.
- **False Positive (FP)**: 3 ulasan negatif salah diprediksi positif.
- **False Negative (FN)**: 1 ulasan positif salah diprediksi negatif.
- **True Positive (TP)**: 29 ulasan positif diprediksi benar.

---

## 5. Analisis Fitur Paling Berpengaruh
- **Indikator Sentimen Positif**: *memukau, mahakarya, spektakuler, brilian, luar biasa, recommended, menyentuh, indah, cerdas*.
- **Indikator Sentimen Negatif**: *membosankan, kaku, murahan, hancur, berantakan, buang-buang waktu, lambat, zonk, klise*.
