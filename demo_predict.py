import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Muat dataset ulasan
df = pd.read_csv("data/movie_reviews.csv")
X = df["text"].values
y = df["label"].values

# 2. Fit TF-IDF dengan stop_words yang sama
indonesian_stopwords = [
    "dan", "yang", "ini", "di", "ke", "dari", "pada", "untuk", "dengan", "adalah",
    "itu", "atau", "dalam", "bisa", "film", "filmnya", "tonton", "menonton", "sangat"
]

vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=500,
    sublinear_tf=True,
    stop_words=indonesian_stopwords
)
X_vec = vec.fit_transform(X)

# 3. Latih XGBoost & LightGBM
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric="logloss")
xgb.fit(X_vec, y)

lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, num_leaves=15, random_state=42, verbose=-1)
lgbm.fit(X_vec, y)

print("="*65)
print("       DEMO INFERENSI MODEL NLP (XGBOOST vs LIGHTGBM)")
print("="*65)

# 4. Uji beberapa ulasan baru di luar dataset
test_sentences = [
    "Alur ceritanya keren banget, efek visual spektakuler dan aktingnya juara!",
    "Film sangat membosankan, akting kaku seperti baca buku dan buang waktu.",
    "Plot twist tak terduga bikin merinding, film terbaik tahun ini wajib nonton!",
    "Jalan cerita hancur berantakan, dialog cringe dan lelucon garing sekali."
]

labels_map = {1: "POSITIF (1)", 0: "NEGATIF (0)"}

for i, text in enumerate(test_sentences, 1):
    t_vec = vec.transform([text])
    xgb_pred = int(xgb.predict(t_vec)[0])
    xgb_prob = xgb.predict_proba(t_vec)[0]

    lgbm_pred = int(lgbm.predict(t_vec)[0])
    lgbm_prob = lgbm.predict_proba(t_vec)[0]

    print(f"\n[Kasus {i}] Ulasan: \"{text}\"")
    print(f"  +-- XGBoost  : {labels_map[xgb_pred]} (Probabilitas: {xgb_prob[xgb_pred]*100:.1f}%)")
    print(f"  +-- LightGBM : {labels_map[lgbm_pred]} (Probabilitas: {lgbm_prob[lgbm_pred]*100:.1f}%)")

print("\n" + "="*65)
print(" Semua model berhasil diuji dan berjalan 100% lancar!")
print("="*65)
