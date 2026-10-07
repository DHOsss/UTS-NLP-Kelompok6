# Klasifikasi Sentimen Ulasan Aplikasi Gojek dengan IndoBERT

Program untuk tugas proyek mata kuliah NLP. Ulasan aplikasi Gojek diambil dari Google Play Store,
lalu diklasifikasikan menjadi positif atau negatif menggunakan **fine-tuning IndoBERT** sebagai
metode satu-satunya (deep learning, sesuai arahan dosen).

## Menjalankan

```
pip install -r requirements.txt
streamlit run app.py
```

Data dan model sudah disertakan, jadi aplikasi bisa langsung dijalankan.
Kalau ingin mengulang dari awal:

```
python scrape.py --app com.gojek.app --count 3000
python train_bert.py     # fine-tuning IndoBERT (butuh GPU, ~1 menit di RTX 4050)
```

`scrape.py` dan `train_bert.py` butuh koneksi internet (`train_bert.py` mengunduh model dasar
IndoBERT ~500 MB dari Hugging Face sekali saja).

## Hasil (280 data uji)

| Metrik | Nilai |
|---|---|
| Akurasi | 89,29% |
| Presisi (macro) | 89,32% |
| Recall (macro) | 89,29% |
| F1-score (macro) | 89,28% |

## Berkas

- `scrape.py` - mengambil ulasan dan memberi label dari rating (1-2 negatif, 4-5 positif)
- `train_bert.py` - fine-tuning IndoBERT, evaluasi, confusion matrix, kurva pelatihan
- `app.py` - antarmuka web Streamlit (uji satu ulasan, uji CSV, tentang model)
- `data/` - ulasan mentah dan daftar data seimbang
- `model/` - model terlatih (`indobert/`), hasil evaluasi, grafik
