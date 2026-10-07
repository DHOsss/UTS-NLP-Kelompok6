"""
Tahap 1 - Pengumpulan data (scraping ulasan Google Play Store).

Mengambil ulasan berbahasa Indonesia untuk satu aplikasi lalu memberi label
sentimen otomatis dari bintang (rating):
    1-2 bintang -> negatif
    4-5 bintang -> positif
    3 bintang   -> dibuang (netral / ambigu)

Jalankan: python scrape.py --app com.gojek.app --count 3000
Output  :  data/ulasan_raw.csv
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # selalu bekerja dari folder code/
import argparse
import pandas as pd
from google_play_scraper import Sort, reviews


def scrape(app_id: str, count: int) -> pd.DataFrame:
    hasil, token = [], None
    while len(hasil) < count:
        batch, token = reviews(
            app_id, lang="id", country="id", sort=Sort.NEWEST,
            count=min(200, count - len(hasil)), continuation_token=token,
        )
        hasil.extend(batch)
        print(f"  terkumpul {len(hasil)} ulasan ...")
        if token is None or not batch:
            break
    df = pd.DataFrame(hasil)[["reviewId", "userName", "score", "at", "content"]]
    df = df.rename(columns={"content": "ulasan", "score": "rating", "at": "tanggal"})
    return df


def beri_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["rating"] != 3].copy()
    df["label"] = df["rating"].apply(lambda s: "positif" if s >= 4 else "negatif")
    df = df.dropna(subset=["ulasan"])
    df = df[df["ulasan"].str.split().str.len() >= 3]      # buang ulasan terlalu pendek
    return df.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="com.gojek.app")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--out", default="data/ulasan_raw.csv")
    a = ap.parse_args()

    print(f"Scraping {a.app} ...")
    df = beri_label(scrape(a.app, a.count))
    df.to_csv(a.out, index=False, encoding="utf-8")
    print(df["label"].value_counts())
    print(f"Tersimpan ke {a.out} ({len(df)} baris)")
