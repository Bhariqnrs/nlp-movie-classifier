import os
import re
import time
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("=== PELATIHAN MODEL NLP UNTUK ULASAN FILM (XGBOOST VS LIGHTGBM) ===")

# 1. Dataset Ulasan Film (Movie Reviews) Bahasa Indonesia & Multi-genre
SAMPLE_MOVIE_REVIEWS = [
    # --- POSITIF (label: 1) ---
    {"text": "Alur ceritanya luar biasa seru, akting pemeran utama sangat memukau dan emosional.", "label": 1, "genre": "Drama"},
    {"text": "Efek visual CGI spektakuler, adegan pertarungan aksi menegangkan dari awal sampai akhir.", "label": 1, "genre": "Action"},
    {"text": "Plot twist di akhir film benar-benar tak terduga dan membuat merinding, mahakarya!", "label": 1, "genre": "Thriller"},
    {"text": "Komedi cerdas yang mengocok perut tanpa terasa garing, humor relate dengan kehidupan.", "label": 1, "genre": "Comedy"},
    {"text": "Film fiksi ilmiah terbaik tahun ini, konsep time travel dan wormhole dieksekusi brilian.", "label": 1, "genre": "Sci-Fi"},
    {"text": "Suasana horor mencekam, jumpscare rapi dan tata suara scoring musik sangat mendukung.", "label": 1, "genre": "Horror"},
    {"text": "Kisah cinta yang manis dan menyentuh hati, chemistry kedua pemeran sangat natural.", "label": 1, "genre": "Romance"},
    {"text": "Sinematografi indah dengan sudut pengambilan gambar kelas dunia, layak menang Oscar!", "label": 1, "genre": "Drama"},
    {"text": "Animasi visual sangat memanjakan mata, pesan moral film ini begitu dalam dan berkesan.", "label": 1, "genre": "Animation"},
    {"text": "Pengembangan karakter sangat matang, sutradara berhasil membawa emosi penonton naik turun.", "label": 1, "genre": "Drama"},
    {"text": "Salah satu film aksi laga terbaik, koreografi bela diri cepat, rapi, dan intens.", "label": 1, "genre": "Action"},
    {"text": "Film dokumenter yang sangat membuka mata dan informatif, penuh data kredibel.", "label": 1, "genre": "Documentary"},
    {"text": "Sangat menghibur dan seru ditonton bareng keluarga di bioskop, wajib nonton!", "label": 1, "genre": "Adventure"},
    {"text": "Soundtrack dan tata musik luar biasa indah, menambah kedalaman rasa di setiap adegan.", "label": 1, "genre": "Drama"},
    {"text": "Misteri pembunuhan tersusun rapi penuh teka-teki cerdas, penonton diajak berpikir.", "label": 1, "genre": "Mystery"},
    {"text": "Akting antagonis begitu berbobot dan menakutkan, penulisan naskah skenario top.", "label": 1, "genre": "Thriller"},
    {"text": "Film petualangan epik dengan latar dunia fantasi megah yang memukau penonton.", "label": 1, "genre": "Fantasy"},
    {"text": "Transisi cerita halus, pacing pas tidak terburu-buru dan akhir cerita sangat memuaskan.", "label": 1, "genre": "Sci-Fi"},
    {"text": "Durasi dua jam terasa cepat karena alur cerita sangat padat dan menarik!", "label": 1, "genre": "Action"},
    {"text": "Sangat direkomendasikan untuk pecinta film berkualitas, 10 dari 10 bintang!", "label": 1, "genre": "Umum"},

    # --- NEGATIF (label: 0) ---
    {"text": "Alur cerita sangat membosankan dan lambat, bikin mengantuk di dalam bioskop.", "label": 0, "genre": "Drama"},
    {"text": "Akting pemeran sangat kaku seperti membaca teks hafalan tanpa ekspresi.", "label": 0, "genre": "Drama"},
    {"text": "Efek visual CGI sangat kasar dan murahan, terlihat tidak nyata dan mengecewakan.", "label": 0, "genre": "Action"},
    {"text": "Plot twist maksa dan ending menggantung tidak masuk akal sama sekali.", "label": 0, "genre": "Thriller"},
    {"text": "Lelucon sangat garing dan cringe, penonton satu studio bioskop hening tanpa tawa.", "label": 0, "genre": "Comedy"},
    {"text": "Konsep cerita fiksi ilmiah kacau, banyak plot hole fatal yang membingungkan.", "label": 0, "genre": "Sci-Fi"},
    {"text": "Bukan horor yang seram melainkan berisik, jumpscare murahan dan jalan cerita hancur.", "label": 0, "genre": "Horror"},
    {"text": "Kisah cinta klise murahan, dialog lebay dan chemistry antar karakter sangat hambar.", "label": 0, "genre": "Romance"},
    {"text": "Pacing cerita sangat berantakan, perpindahan adegan meloncat-loncat membingungkan.", "label": 0, "genre": "Drama"},
    {"text": "Kualitas naskah sangat buruk, dialog kaku tidak bernyawa buang-buang waktu.", "label": 0, "genre": "Drama"},
    {"text": "Adegan aksi kamera goyang berlebihan bikin pusing mual, koreografi tidak jelas.", "label": 0, "genre": "Action"},
    {"text": "Karakter utama sangat menyebalkan dan tidak konsisten, jalan cerita monoton.", "label": 0, "genre": "Adventure"},
    {"text": "Musik latar terlalu keras menutupi suara dialog percakapan pemain, tata audio payah.", "label": 0, "genre": "Horror"},
    {"text": "Ekspektasi tinggi tapi hasilnya zonk, sangat mengecewakan dan buang-buang tiket.", "label": 0, "genre": "Umum"},
    {"text": "Durasi terasa sangat lama dan membosankan, ingin segera keluar studio bioskop.", "label": 0, "genre": "Drama"},
    {"text": "Penjahatnya konyol tanpa motif jelas, pertarungan akhir antiklimaks yang payah.", "label": 0, "genre": "Action"},
    {"text": "Animasi kaku dan warna redup tidak menarik, cerita membingungkan anak-anak.", "label": 0, "genre": "Animation"},
    {"text": "Film terburuk yang pernah saya tonton tahun ini, sangat tidak direkomendasikan!", "label": 0, "genre": "Umum"},
    {"text": "Ending film sangat merusak keseluruhan cerita, kesal sekali menontonnya.", "label": 0, "genre": "Mystery"},
    {"text": "Sutradara gagal mengeksekusi premis yang sebenarnya punya potensi bagus.", "label": 0, "genre": "Sci-Fi"}
]

# Perluas data variasi ulasan agar representatif untuk pelatihan machine learning
EXPANDED_REVIEWS = []
for item in SAMPLE_MOVIE_REVIEWS:
    EXPANDED_REVIEWS.append(item)
    if item["label"] == 1:
        EXPANDED_REVIEWS.append({
            "text": "Keren banget! " + item["text"] + " Sangat puas menontonnya.",
            "label": 1,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Pilihan tepat, " + item["text"].lower() + " Recommended.",
            "label": 1,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Bagus sekali filmnya, " + item["text"].lower(),
            "label": 1,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Karya spektakuler. " + item["text"],
            "label": 1,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Wajib nonton ulang! " + item["text"],
            "label": 1,
            "genre": item["genre"]
        })
    else:
        EXPANDED_REVIEWS.append({
            "text": "Kecewa parah! " + item["text"] + " Menyesal beli tiketnya.",
            "label": 0,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Jangan tonton film ini, " + item["text"].lower() + " Hancur total.",
            "label": 0,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Sangat buruk kualitasnya, " + item["text"].lower(),
            "label": 0,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Zonk banget. " + item["text"],
            "label": 0,
            "genre": item["genre"]
        })
        EXPANDED_REVIEWS.append({
            "text": "Rugi waktu nonton, " + item["text"],
            "label": 0,
            "genre": item["genre"]
        })

df_movies = pd.DataFrame(EXPANDED_REVIEWS)

# 2. Tokenizer Regex (Persis referensi user)
def tokens(text):
    return re.findall(r"[a-zA-ZÀ-ÿ]+", str(text).lower())

df_movies["n_char"] = df_movies["text"].str.len()
df_movies["n_token"] = df_movies["text"].map(lambda x: len(tokens(x)))

print("Total Data Ulasan Film:", len(df_movies))
print("Distribusi Kelas Sentimen:", df_movies["label"].value_counts().to_dict())

all_tokens = [t for x in df_movies["text"] for t in tokens(x)]
vocab_counter = Counter(all_tokens)
print("Total Token Kata:", len(all_tokens))
print("Kosakata Unik:", len(vocab_counter))
print("Top 10 Kata:", vocab_counter.most_common(10))

# Simpan CSV ulasan film
os.makedirs("data", exist_ok=True)
csv_movie_path = os.path.join("data", "movie_reviews.csv")
df_movies.to_csv(csv_movie_path, index=False, encoding="utf-8")

# 3. Train-Test Split (Stratified 75:25)
X = df_movies["text"].values
y = df_movies["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"Data Latih: {len(X_train)} | Data Uji: {len(X_test)}")

# 4. TF-IDF Ekstraksi Fitur (Unigram & Bigram dengan Stop Words & Sublinear TF)
indonesian_stopwords = [
    "dan", "yang", "ini", "di", "ke", "dari", "pada", "untuk", "dengan", "adalah",
    "itu", "atau", "dalam", "bisa", "film", "filmnya", "tonton", "menonton", "sangat"
]

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_features=500,
    sublinear_tf=True,
    stop_words=indonesian_stopwords
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
feature_names = tfidf.get_feature_names_out().tolist()

# 5. Pelatihan Model (XGBoost, LightGBM, Baseline Logistic Regression)
models = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
}

results = {}
feature_importances = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_vec, y_train)
    t_train = round(time.time() - t0, 4)

    # Inferensi
    t_inf_start = time.time()
    preds = model.predict(X_test_vec)
    t_latency = round((time.time() - t_inf_start) / len(X_test) * 1000, 3)

    acc = float(accuracy_score(y_test, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    results[name] = {
        "accuracy": round(acc * 100, 2),
        "macro_precision": round(float(prec) * 100, 2),
        "macro_recall": round(float(rec) * 100, 2),
        "macro_f1": round(float(f1) * 100, 2),
        "train_time_sec": t_train,
        "inference_ms_per_sample": t_latency,
        "confusion_matrix": cm
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy: {results[name]['accuracy']}% | Macro F1: {results[name]['macro_f1']}%")
    print(f"Waktu Latih: {t_train}s | Latensi: {t_latency} ms/item")
    print("Confusion Matrix:", cm)

    if hasattr(model, "feature_importances_"):
        top_idx = np.argsort(model.feature_importances_)[::-1][:15]
        feature_importances[name] = [
            {"word": feature_names[i], "score": round(float(model.feature_importances_[i]), 5)}
            for i in top_idx if model.feature_importances_[i] > 0
        ]

# Bangun Kamus Bobot Sentimen Film untuk UI Web Interaktif
sentiment_lexicon = {}
for word in feature_names:
    pos_c = sum(1 for item in EXPANDED_REVIEWS if word in item["text"].lower() and item["label"] == 1)
    neg_c = sum(1 for item in EXPANDED_REVIEWS if word in item["text"].lower() and item["label"] == 0)
    sentiment_lexicon[word] = {
        "pos_freq": pos_c,
        "neg_freq": neg_c,
        "polarity": round((pos_c - neg_c) / max(1, pos_c + neg_c), 3)
    }

export_json = {
    "metadata": {
        "title": "Analisis Sentimen Ulasan Film Menggunakan XGBoost dan LightGBM berbasis TF-IDF",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset": len(df_movies),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocab_size": len(feature_names),
        "ngram_range": [1, 2],
        "split_ratio": "75:25 Stratified"
    },
    "results": results,
    "feature_importances": feature_importances,
    "sentiment_lexicon": sentiment_lexicon
}

out_path = os.path.join("data", "movie_nlp_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(export_json, f, ensure_ascii=False, indent=2)

print(f"\nHasil berhasil diekspor ke {out_path}")
