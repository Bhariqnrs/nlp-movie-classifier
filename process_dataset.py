import pandas as pd
import json
import re

print("Memulai pemrosesan dataset...")

# 1. Load movies metadata
movies_meta_path = r"c:\Users\HANSEL\Downloads\movies.csv"
print(f"Membaca {movies_meta_path}...")
df_meta = pd.read_csv(movies_meta_path)
print(f"Total film di metadata: {len(df_meta)}")

# 2. Process ratings in chunks from 'Dataset Movie len 32m.csv'
ratings_path = r"c:\Users\HANSEL\Downloads\nlp-movie-classifier\data\Dataset Movie len 32m.csv"
print(f"Membaca ratings dari {ratings_path}...")

rating_stats = {} # movieId -> [sum_rating, count]

chunk_size = 500000
for i, chunk in enumerate(pd.read_csv(ratings_path, chunksize=chunk_size, usecols=['movieId', 'systemPredictRating'])):
    # Group by movieId in chunk
    grp = chunk.groupby('movieId')['systemPredictRating'].agg(['sum', 'count'])
    for mid, row in grp.iterrows():
        if mid not in rating_stats:
            rating_stats[mid] = [row['sum'], row['count']]
        else:
            rating_stats[mid][0] += row['sum']
            rating_stats[mid][1] += row['count']
    print(f"Chunk {i+1} selesai...")

print(f"Total film yang memiliki rating: {len(rating_stats)}")

# 3. Gabungkan rating dengan metadata
records = []
for mid, (r_sum, count) in rating_stats.items():
    if count >= 3: # minimal 3 rating agar valid
        avg_rating = r_sum / count
        # Skala rating di dataset biasanya 0.5 - 5.0, kita konversi ke skala 1 - 10
        # Jika rata-rata <= 5, kalikan 2 jadi skala 10
        scaled_rating = round(avg_rating * 2, 1) if avg_rating <= 5.0 else round(avg_rating, 1)
        scaled_rating = min(10.0, max(1.0, scaled_rating))
        records.append({
            'movieId': mid,
            'rating': scaled_rating,
            'votes': count
        })

df_ratings = pd.DataFrame(records)
df_merged = pd.merge(df_ratings, df_meta, on='movieId', how='inner')

# Urutkan berdasarkan rating tertinggi dan jumlah votes
df_merged = df_merged.sort_values(by=['rating', 'votes'], ascending=[False, False])

# Ambil top 500 film terbaik agar performa web tetap super cepat dan ringan
df_top = df_merged.head(500)

result = []
year_regex = re.compile(r"\((\d{4})\)$")

for _, row in df_top.iterrows():
    title_raw = str(row['title']).strip()
    year = None
    match = year_regex.search(title_raw)
    clean_title = title_raw
    if match:
        year = int(match.group(1))
        clean_title = title_raw[:match.start()].strip()

    genre_str = str(row['genres'])
    if genre_str == '(no genres listed)':
        genres = ['Umum']
    else:
        genres = [g.strip() for g in genre_str.split('|') if g.strip()]

    result.append({
        'id': int(row['movieId']),
        'title': clean_title,
        'genre': genres,
        'rating': float(row['rating']),
        'year': year,
        'director': '-',
        'duration': None,
        'description': f"Film {clean_title} ({year or ''}) dalam kategori {', '.join(genres)}. Dinilai berdasarkan {int(row['votes'])} review pengguna.",
        'poster': ''
    })

output_path = r"c:\Users\HANSEL\Downloads\nlp-movie-classifier\data\movies.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Sukses! {len(result)} film berhasil diproses dan disimpan ke {output_path}")
