# Sentimen Ulasan Aplikasi Gojek di Google Play Store Berbasis IndoBERT

Tugas Proyek UTS - Natural Language Processing (NLP)

Sistem klasifikasi sentimen untuk ulasan aplikasi Gojek di Google Play Store berbahasa Indonesia.
**Metode satu-satunya: fine-tuning IndoBERT** (indobenchmark/indobert-base-p1, deep learning) pada
1.120 ulasan latih, diuji pada 280 data uji. Teks ulasan mentah diproses langsung tanpa stemming
maupun stopword removal karena tokenisasi subword IndoBERT mampu menangani kata tidak baku.

## Hasil

| Metrik | Nilai |
|---|---|
| Akurasi | **89,29%** |
| Presisi (macro) | 89,32% |
| Recall (macro) | 89,29% |
| F1-score (macro) | 89,28% |

Sebagai pembanding dari literatur: IndoBERT 89% pada ulasan ChatGPT Play Store (Mursidah dkk.,
2025) dan sekitar 89% metode klasik pada ulasan Gojek (Rousyati dkk., 2026).

## Alur Sistem

Scraping 3.000 ulasan (google-play-scraper) → pelabelan otomatis dari rating (1-2 negatif, 4-5
positif) → penyeimbangan kelas (700 + 700) → split 80:20 (seed tetap) → tokenisasi subword
WordPiece maks 128 token (teks mentah) → fine-tuning IndoBERT (4 epok, batch 16, lr 2e-5) →
evaluasi (akurasi, precision, recall, F1, confusion matrix) → antarmuka web Streamlit.

## Cara Menjalankan

```
cd code
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi langsung bisa jalan. Untuk melatih ulang model (regenerasi bobot):

```
python train_bert.py
```

Mengulang sepenuhnya dari awal (opsional):

```
python scrape.py --app com.gojek.app --count 3000
python train_bert.py
```

## Struktur Repositori

```
code/                  program lengkap
├── app.py             antarmuka web Streamlit
├── scrape.py          pengumpulan data ulasan + pelabelan otomatis
├── train_bert.py      fine-tuning IndoBERT (metode utama) + evaluasi
├── data/              2.035 ulasan mentah + daftar 1.400 ulasan seimbang
└── model/             hasil evaluasi, grafik, model tersimpan
laporan/               tulisan ilmiah Bab 1-3 (PDF)
manual/                manual penjelasan program (PDF)
perhitungan/           contoh perhitungan manual lapisan keluaran & evaluasi (PDF)
rancangan/             flowchart sistem & wireframe web/mobile
```

## Referensi Utama

1. B. Wilie et al., "IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural
   Language Understanding," AACL-IJCNLP 2020. (model dasar IndoBERT)
2. I. Mursidah et al., "Klasifikasi Sentimen Google Play Store Aplikasi ChatGPT Berbahasa
   Indonesia Berbasis IndoBERT," Jurnal Minfo Polgan 14(2), 2025.
3. Rousyati, D. Pratmanto, Fandhilah, "Perbandingan Algoritma Naive Bayes dan K-Nearest
   Neighbors dalam Analisis Sentimen Ulasan Aplikasi Gojek," METHOMIKA 10(1), 2026.

## Tim

| Anggota | NIM | Peran |
|---|---|---|
| Dave Famataronia Gulo | 535250042 | Data & referensi |
| Willwen Adinata | 535250051 | Data & evaluasi |
| Aldho Prahauga | 535250153 | Model utama & UI/UX |

Universitas Tarumanagara - Program Studi Teknik Informatika, 2026.
