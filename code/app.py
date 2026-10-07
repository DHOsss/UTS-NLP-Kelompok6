"""
Antarmuka web sederhana (Streamlit) untuk klasifikasi sentimen ulasan.

Model: IndoBERT hasil fine-tuning (model/indobert) - satu-satunya metode yang digunakan.

Jalankan:  streamlit run app.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # selalu bekerja dari folder code/

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="Sentimen Ulasan Aplikasi", page_icon=":speech_balloon:", layout="wide")


@st.cache_resource
def muat_bert():
    tokenizer = AutoTokenizer.from_pretrained("model/indobert")
    # muat sebagai float32 agar kompatibel dengan CPU (bobot tersimpan bfloat16)
    model = AutoModelForSequenceClassification.from_pretrained(
        "model/indobert", dtype=torch.float32)
    model.eval()
    return tokenizer, model


def prediksi(teks_list):
    """Klasifikasi dengan IndoBERT langsung pada teks mentah (tanpa preprocessing berat)."""
    tokenizer, model = muat_bert()
    probs = []
    with torch.no_grad():
        for i in range(0, len(teks_list), 64):
            enc = tokenizer([str(t) for t in teks_list[i:i + 64]], truncation=True,
                            padding=True, max_length=128, return_tensors="pt")
            logit = model(**enc).logits
            probs.append(torch.softmax(logit, dim=1).float())
    return torch.cat(probs).numpy(), ["negatif", "positif"]


st.title("Klasifikasi Sentimen Ulasan Aplikasi Google Play")
st.caption("Model utama: IndoBERT (fine-tuned) | Tugas Proyek NLP - UTS")

tab1, tab2, tab3 = st.tabs(["Uji satu ulasan", "Uji banyak ulasan (CSV)", "Tentang model"])

with tab1:
    teks = st.text_area("Tulis ulasan di sini", "Aplikasinya gak bagus bgt, sering error dan lemot", height=120)
    if st.button("Klasifikasikan", type="primary"):
        prob, kelas = prediksi([teks])
        prob, kelas = prob[0], kelas
        label = kelas[int(np.argmax(prob))]
        warna = "green" if label == "positif" else "red"
        st.markdown(f"### Hasil: :{warna}[{label.upper()}]  (keyakinan {prob.max():.1%})")
        st.progress(float(prob[kelas.index('positif')]), text=f"P(positif) = {prob[kelas.index('positif')]:.3f}")
        with st.expander("Lihat tokenisasi subword IndoBERT"):
            tokenizer, _ = muat_bert()
            tokens = tokenizer.tokenize(str(teks))
            st.write(f"**{len(tokens)} token**: `{' | '.join(tokens)}`")
        st.info("IndoBERT memproses teks mentah langsung menjadi token subword, "
                "sehingga tidak memerlukan stemming maupun stopword removal.")

with tab2:
    st.write("Unggah CSV dengan kolom **ulasan**.")
    f = st.file_uploader("Pilih file", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        prob, kelas = prediksi(df["ulasan"].tolist())
        df["prediksi"] = [kelas[i] for i in prob.argmax(axis=1)]
        df["p_positif"] = prob[:, kelas.index("positif")].round(3)
        c1, c2 = st.columns([1, 2])
        c1.metric("Jumlah ulasan", len(df))
        c1.bar_chart(df["prediksi"].value_counts())
        c2.dataframe(df[["ulasan", "prediksi", "p_positif"]], use_container_width=True)
        st.download_button("Unduh hasil", df.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv")

with tab3:
    try:
        st.code(open("model/hasil_evaluasi_bert.txt", encoding="utf-8").read())
        c1, c2 = st.columns(2)
        c1.image("model/confusion_matrix_bert.png", caption="Confusion matrix IndoBERT")
        c2.image("model/kurva_pelatihan_bert.png", caption="Kurva pelatihan dan akurasi uji per epok")
    except FileNotFoundError:
        st.warning("Jalankan train_bert.py terlebih dahulu.")
