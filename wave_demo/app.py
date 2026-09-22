# WAVE demo app (Streamlit).
#
# Interactive, self-contained demo of the WAVE architecture
# (https://github.com/ZJUFanLab/WAVE): a VAE over baseline gene
# expression, a small MLP over drug fingerprints, and a fusion head
# that predicts the perturbation delta. Because the real CMap/L1000
# training data isn't available in this environment, the app trains
# on a small synthetic dataset (see data.py) purely to demonstrate
# the pipeline end to end. Predictions are NOT real biological
# results.

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import torch.optim as optim

from data import FP_BITS, N_GENES, PRESET_DRUGS, build_synthetic_dataset, morgan_fp, random_baseline
from model import WAVE

st.set_page_config(page_title="WAVE Demo", page_icon="🧬", layout="wide")


def loss_fct(pred, target, mu, logvar, alpha=1.0, beta=0.01):
    mse_loss = F.mse_loss(pred, target)

    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred_mean * target_mean, dim=1)
    denominator = torch.sqrt(torch.sum(pred_mean ** 2, dim=1) * torch.sum(target_mean ** 2, dim=1)) + 1e-8
    pearson_loss = 1 - torch.mean(numerator / denominator)

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)

    return mse_loss + alpha * pearson_loss + beta * kld_loss


def train_demo_model(epochs, batch_size, seed, progress_bar, status_text):
    torch.manual_seed(seed)
    unpert, drug_fp, pert, effect_vectors = build_synthetic_dataset(seed=seed)

    model = WAVE()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    unpert_t = torch.tensor(unpert)
    drug_fp_t = torch.tensor(drug_fp)
    pert_t = torch.tensor(pert)
    n = unpert_t.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred, mu, logvar, _ = model(unpert_t[idx], drug_fp_t[idx])
            loss = loss_fct(pred, pert_t[idx], mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.3f}")

    model.eval()
    return model, effect_vectors


st.title("🧬 WAVE Demo — İlaç Kaynaklı Gen Ekspresyonu Tahmini")

st.markdown(
    """
Bu uygulama [ZJUFanLab/WAVE](https://github.com/ZJUFanLab/WAVE) modelinin mimarisini
(GeneVAE + DrugNN + GeneDrugFusion) temel alan **öğretici bir gösterim**dir.

> ⚠️ **Önemli:** Gerçek CMap/L1000 eğitim verisi veya önceden eğitilmiş ağırlıklar bu ortamda
> mevcut değil. Aşağıdaki "Modeli Eğit" adımı, birkaç örnek ilaç için üretilmiş **sentetik**
> bir veri kümesi üzerinde çalışır. Amaç, WAVE mimarisinin ve eğitim döngüsünün gerçekten
> nasıl çalıştığını uçtan uca göstermektir — tahminler **gerçek biyolojik sonuçlar değildir**.
"""
)

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.effect_vectors = None

with st.sidebar:
    st.header("1) Modeli Eğit")
    epochs = st.slider("Epoch sayısı", min_value=5, max_value=100, value=40, step=5)
    seed = st.number_input("Rastgelelik tohumu (seed)", value=1, step=1)

    train_clicked = st.button("Modeli Eğit (sentetik veri)", type="primary")

    if train_clicked:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        model, effect_vectors = train_demo_model(
            epochs=int(epochs), batch_size=32, seed=int(seed),
            progress_bar=progress_bar, status_text=status_text,
        )
        st.session_state.model = model
        st.session_state.effect_vectors = effect_vectors
        st.success("Eğitim tamamlandı.")

    st.divider()
    st.caption(
        "Mimari: 978 gen → GeneVAE (latent=128) · SMILES → 2048-bit Morgan parmak izi → "
        "DrugNN (128) · GeneDrugFusion, VAE çıktısına eklenecek delta ekspresyonu üretir."
    )

if st.session_state.model is None:
    st.info("Tahmin yapmak için önce soldaki panelden modeli eğitin.")
    st.stop()

st.header("2) İlaç ve Baseline Gen Ekspresyonu Seç")

col1, col2 = st.columns(2)

with col1:
    drug_mode = st.radio("İlaç kaynağı", ["Hazır ilaç", "Kendi SMILES'im"], horizontal=True)
    if drug_mode == "Hazır ilaç":
        drug_name = st.selectbox("İlaç", list(PRESET_DRUGS.keys()))
        smiles = PRESET_DRUGS[drug_name]
        st.code(smiles, language="text")
        known_effect = st.session_state.effect_vectors.get(drug_name)
    else:
        smiles = st.text_input("SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
        known_effect = None

    fp = morgan_fp(smiles) if smiles else None
    if smiles and fp is None:
        st.error("Geçersiz SMILES — RDKit bu molekülü ayrıştıramadı.")

with col2:
    if "baseline" not in st.session_state:
        st.session_state.baseline = random_baseline(1, seed=int(seed))[0]

    if st.button("🎲 Rastgele baseline profili oluştur"):
        st.session_state.baseline = random_baseline(1, seed=np.random.randint(0, 1_000_000))[0]

    baseline = st.session_state.baseline
    st.caption(f"Baseline profili: {N_GENES} sentetik gen, ortalama={baseline.mean():.2f}, std={baseline.std():.2f}")
    st.bar_chart(pd.DataFrame({"baseline": baseline[:40]}))
    st.caption("İlk 40 sentetik gen gösteriliyor (gerçek gen sembolleri değildir).")

st.header("3) Tahmin")

predict_clicked = st.button("Tahmin Et", type="primary", disabled=(fp is None))

if predict_clicked and fp is not None:
    model = st.session_state.model
    with torch.no_grad():
        baseline_t = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)
        fp_t = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
        pred, mu, logvar, delta = model(baseline_t, fp_t)

    pred = pred.squeeze(0).numpy()
    delta = delta.squeeze(0).numpy()

    m1, m2, m3 = st.columns(3)
    m1.metric("Tahmini ortalama ekspresyon", f"{pred.mean():.3f}")
    m2.metric("Ortalama |delta|", f"{np.abs(delta).mean():.3f}")

    if known_effect is not None:
        corr = np.corrcoef(delta, known_effect)[0, 1]
        m3.metric("Sentetik gerçek etkiyle Pearson r", f"{corr:.3f}")
        st.caption(
            "Bu değer, model eğitim sırasında bu ilacın sentetik 'gerçek' etki vektörünü "
            "ne kadar öğrenmiş olduğunun bir doğrulamasıdır (1'e yakınsa iyi öğrenmiştir)."
        )
    else:
        m3.metric("Sentetik gerçek etkiyle Pearson r", "N/A (bilinmeyen ilaç)")

    df = pd.DataFrame({
        "baseline": baseline,
        "tahmin (predicted)": pred,
        "delta": delta,
    })
    st.subheader("Baseline vs Tahmin (ilk 40 sentetik gen)")
    st.line_chart(df[["baseline", "tahmin (predicted)"]].iloc[:40])

    st.subheader("En çok değişen 15 sentetik gen (|delta|)")
    top_idx = np.argsort(-np.abs(delta))[:15]
    top_df = pd.DataFrame({
        "gen": [f"Gene_{i}" for i in top_idx],
        "delta": delta[top_idx],
    }).set_index("gen")
    st.bar_chart(top_df)

    with st.expander("Ham veriyi indir"):
        st.download_button(
            "Tahmini CSV olarak indir",
            df.to_csv(index_label="gene_index"),
            file_name="wave_demo_prediction.csv",
            mime="text/csv",
        )
