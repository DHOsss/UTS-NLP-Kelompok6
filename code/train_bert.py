"""
Tahap 3B - Fine-tuning IndoBERT untuk klasifikasi sentimen (model utama),
TF-IDF + Naive Bayes pada train.py berperan sebagai baseline pembanding.

Jalankan:  python train_bert.py
Input   :  data/ulasan_raw.csv  +  baris data yang sama dengan ulasan_bersih.csv
Output  :  model/indobert/              (model terlatih + tokenizer)
           model/hasil_evaluasi_bert.txt(akurasi, precision, recall, F1)
           model/confusion_matrix_bert.png
           model/kurva_pelatihan_bert.png

Catatan penting untuk laporan:
- IndoBERT memakai TEKS MENTAH (tanpa stemming/stopword removal) karena model
  transformer memproses kata menjadi subword dan memahami konteks kalimat;
  preprocessing berat justru membuang informasi.
- Data latih/uji SAMA PERSIS dengan baseline Naive Bayes (seed sama) sehingga
  hasil keduanya dapat dibandingkan secara adil di satu tabel.
"""
import json
import os
import random
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # selalu bekerja dari folder code/

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                              classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

SEED = 42
NAMA_MODEL = "indobenchmark/indobert-base-p1"   # model bahasa Indonesia (IndoNLU)
PANJANG_MAKS = 128
EPOK = 4
BATCH = 16
LR = 2e-5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); set_seed(SEED)
os.makedirs("model", exist_ok=True)

# ---------------------------------------------------------------- 1. data
# SAMA PERSIS dengan baseline NB: baris & URUTAN mengikuti ulasan_bersih.csv
# (urutan itulah yang dipakai train.py saat split), sehingga data uji identik 100%.
bersih = pd.read_csv("data/ulasan_bersih.csv")[["reviewId", "label"]]
raw = pd.read_csv("data/ulasan_raw.csv").set_index("reviewId")
df = bersih.copy()
df["ulasan"] = df["reviewId"].map(raw["ulasan"])
assert df["ulasan"].notna().all(), "ada reviewId yang tidak cocok"
print(f"Data: {len(df)} ulasan (urutan & baris identik dengan baseline Naive Bayes)")
print(df["label"].value_counts(), "\n")

# split 80:20 identik dengan train.py
id_latih, id_uji = train_test_split(
    df["reviewId"], test_size=0.2, random_state=SEED, stratify=df["label"])
teks_latih = df[df["reviewId"].isin(id_latih)]["ulasan"].tolist()
teks_uji   = df[df["reviewId"].isin(id_uji)]["ulasan"].tolist()
y_latih    = df[df["reviewId"].isin(id_latih)]["label"].map({"negatif": 0, "positif": 1}).tolist()
y_uji      = df[df["reviewId"].isin(id_uji)]["label"].map({"negatif": 0, "positif": 1}).tolist()
print(f"Latih: {len(teks_latih)}  Uji: {len(teks_uji)}\n")

# ---------------------------------------------------------------- 2. tokenisasi
print(f"Memuat model {NAMA_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(NAMA_MODEL)

class DatasetUlasan(Dataset):
    def __init__(self, teks, label):
        self.enc = tokenizer(teks, truncation=True, padding="max_length",
                             max_length=PANJANG_MAKS, return_tensors=None)
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.label[i])
        return item

model = AutoModelForSequenceClassification.from_pretrained(
    NAMA_MODEL, num_labels=2,
    id2label={0: "negatif", 1: "positif"}, label2id={"negatif": 0, "positif": 1})

# ---------------------------------------------------------------- 3. fine-tuning
def hitung_metrik(pred):
    logit, label = pred
    y_pred = np.argmax(logit, axis=1)
    return {"akurasi": accuracy_score(label, y_pred),
            "presisi": precision_score(label, y_pred, average="macro"),
            "recall": recall_score(label, y_pred, average="macro"),
            "f1": f1_score(label, y_pred, average="macro")}

args = TrainingArguments(
    output_dir="model/_cekpt",
    num_train_epochs=EPOK,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=32,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=14,          # ~10% dari 140 langkah (2 epok hangat)
    bf16=torch.cuda.is_bf16_supported(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="akurasi",
    greater_is_better=True,
    save_total_limit=1,
    logging_steps=20,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(model=model, args=args,
                  train_dataset=DatasetUlasan(teks_latih, y_latih),
                  eval_dataset=DatasetUlasan(teks_uji, y_uji),
                  processing_class=tokenizer,
                  compute_metrics=hitung_metrik)

t0 = time.time()
hasil = trainer.train()
print(f"\nFine-tuning selesai dalam {(time.time()-t0)/60:.1f} menit")

# ---------------------------------------------------------------- 4. evaluasi akhir
pred_logit = trainer.predict(DatasetUlasan(teks_uji, y_uji)).predictions
y_pred = np.argmax(pred_logit, axis=1)

akurasi = accuracy_score(y_uji, y_pred)
laporan = (
    "=== IndoBERT (fine-tuned, model utama) ===\n"
    f"Akurasi : {akurasi:.4f}\n"
    f"Presisi : {precision_score(y_uji, y_pred, average='macro'):.4f}\n"
    f"Recall  : {recall_score(y_uji, y_pred, average='macro'):.4f}\n"
    f"F1      : {f1_score(y_uji, y_pred, average='macro'):.4f}\n\n"
    "Confusion matrix (baris = aktual, kolom = prediksi) [negatif, positif]:\n"
    f"{confusion_matrix(y_uji, y_pred)}\n\n"
    f"{classification_report(y_uji, y_pred, target_names=['negatif', 'positif'], digits=4)}\n"
)
print(laporan)

# bandingkan dengan baseline Naive Bayes
teks_nb = open("model/hasil_evaluasi.txt", encoding="utf-8").read()
import re
akurasi_nb = float(re.search(r"=== Multinomial Naive Bayes ===\nAkurasi : ([\d.]+)", teks_nb).group(1))
with open("model/hasil_evaluasi_bert.txt", "w", encoding="utf-8") as f:
    f.write(f"Jumlah data: {len(df)} | latih {len(teks_latih)} | uji {len(teks_uji)}"
            f" | teks mentah, panjang maks {PANJANG_MAKS} token\n")
    f.write(f"Model dasar: {NAMA_MODEL} | epok {EPOK} | batch {BATCH} | lr {LR}\n\n")
    f.write(laporan)
    f.write(f"PERBANDINGAN (data & split identik):\n"
            f"  Naive Bayes (baseline) : {akurasi_nb*100:.2f}%\n"
            f"  IndoBERT (model utama) : {akurasi*100:.2f}%\n")

# ---------------------------------------------------------------- 5. simpan & visualisasi
trainer.save_model("model/indobert")
tokenizer.save_pretrained("model/indobert")

cm = confusion_matrix(y_uji, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay(cm, display_labels=["negatif", "positif"]).plot(
    ax=ax, colorbar=False, cmap="Greens")
ax.set_title(f"Confusion Matrix - IndoBERT (akurasi {akurasi*100:.2f}%)")
fig.tight_layout(); fig.savefig("model/confusion_matrix_bert.png", dpi=150)

riwayat = trainer.state.log_history
epok_eval = [h["epoch"] for h in riwayat if "eval_akurasi" in h]
ak_eval   = [h["eval_akurasi"] for h in riwayat if "eval_akurasi" in h]
langkah   = [h["epoch"] for h in riwayat if "loss" in h]
loss      = [h["loss"] for h in riwayat if "loss" in h]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(langkah, loss, "o-", color="#457b9d"); axes[0].set_xlabel("epok")
axes[0].set_ylabel("loss latih"); axes[0].set_title("Kurva loss pelatihan IndoBERT")
axes[1].plot(epok_eval, [a*100 for a in ak_eval], "s-", color="#2a9d8f")
axes[1].set_xlabel("epok"); axes[1].set_ylabel("akurasi uji (%)")
axes[1].set_title("Akurasi uji per epok"); axes[1].set_ylim(50, 100)
fig.tight_layout(); fig.savefig("model/kurva_pelatihan_bert.png", dpi=150)

print("Model IndoBERT & grafik tersimpan di model/indobert, model/confusion_matrix_bert.png, model/kurva_pelatihan_bert.png")
