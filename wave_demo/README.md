# WAVE Demo

[ZJUFanLab/WAVE](https://github.com/ZJUFanLab/WAVE) modelinin mimarisini (GeneVAE +
DrugNN + GeneDrugFusion) temel alan, çalıştırılabilir bir Streamlit gösterim uygulaması.

WAVE, bir ilacın (SMILES ile temsil edilir) hücrenin baseline (tedavi öncesi) gen
ekspresyon profilini nasıl değiştireceğini tahmin etmeye çalışan bir sinir ağı modelidir.

## ⚠️ Önemli sınırlama

Bu ortamda gerçek CMap/L1000 eğitim verisi veya WAVE için önceden eğitilmiş ağırlıklar
mevcut değil. Bu yüzden uygulama, birkaç örnek ilaç (`data.py` içindeki `PRESET_DRUGS`)
için üretilmiş **sentetik** bir veri kümesi üzerinde küçük bir model eğitir: her ilaca
sabit, rastgele bir "etki vektörü" atanır ve model bunu baseline + gürültüden ayırt
etmeyi öğrenir. Amaç, WAVE mimarisinin ve eğitim döngüsünün gerçekten nasıl çalıştığını
uçtan uca göstermektir. **Üretilen tahminler gerçek biyolojik sonuçlar değildir.**

Gerçek verilerle kullanmak isterseniz orijinal repodaki `wave/train.py` ve
`wave/load_dataset.py` dosyalarına bakın; gerçek `.h5ad` (AnnData) formatında eğitim
verisi gerektirir.

## Kurulum ve çalıştırma

```bash
cd wave_demo
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda açılan sayfada:

1. Soldaki panelden **"Modeli Eğit (sentetik veri)"** butonuna basıp modelin sentetik
   veri üzerinde eğitilmesini bekleyin (birkaç saniye - bir dakika, CPU'ya bağlı).
2. Hazır bir ilaç seçin veya kendi SMILES'inizi girin.
3. İsterseniz rastgele bir baseline gen ekspresyon profili oluşturun.
4. **"Tahmin Et"** ile modelin öngördüğü ekspresyon profilini, en çok değişen genleri
   ve (bilinen ilaçlar için) modelin sentetik gerçek etkiyi ne kadar doğru öğrendiğini
   gösteren Pearson korelasyonunu görün.

## Dosyalar

- `model.py` — WAVE mimarisi (GeneVAE, DrugNN, GeneDrugFusion), orijinal WAVE reposundan
  (MIT lisans) uyarlanmıştır.
- `data.py` — Sentetik veri üretimi ve SMILES → Morgan fingerprint dönüşümü.
- `app.py` — Streamlit arayüzü: eğitim, tahmin ve görselleştirme.

## Lisans / Atıf

Model mimarisi [ZJUFanLab/WAVE](https://github.com/ZJUFanLab/WAVE) reposundan MIT
lisansı altında uyarlanmıştır (Copyright (c) 2026 Development Team).
