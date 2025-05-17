
# 🎵 Music Genre Classification with Machine Learning

Bu proje, bir şarkının temel müzik özelliklerine (danceability, energy, tempo vb.) göre hangi **müzik türüne** ait olduğunu tahmin etmeyi amaçlayan bir makine öğrenmesi modelini içerir. Model, **Spotify şarkı verileri** ile eğitilmiş ve `Random Forest` algoritması ile gerçekleştirilmiştir.

## 🚀 Özellikler

- 🎧 Spotify veri kümesi ile eğitilmiş model
- 🎯 SMOTE ile dengesiz sınıfların dengelenmesi
- 🌲 Random Forest sınıflandırıcısı
- 🧪 Doğruluk, Precision, Recall ve F1-score çıktıları
- 📈 Scikit-learn kullanarak kapsamlı değerlendirme raporu
- 🎹 Kullanıcının girdiği özelliklere göre canlı tahmin imkanı

## 🧠 Kullanılan Teknolojiler

- Python 3
- Pandas
- NumPy
- Scikit-learn
- imbalanced-learn (SMOTE için)
- Matplotlib (isteğe bağlı görselleştirme için eklenebilir)

## 📁 Veri Kümesi

Spotify şarkı verileri içeren `spotify_songs.csv` adlı dosya `archive/` klasörü altında yer almalıdır. Bu dosyada her bir şarkıya ait aşağıdaki özellikler bulunmaktadır:

- `danceability`
- `energy`
- `tempo`
- `loudness`
- `valence`
- `acousticness`
- `instrumentalness`
- `speechiness`
- `duration_ms`
- `playlist_genre` (hedef değişken)

## ⚙️ Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
````

2. Python dosyasını çalıştırın:

```bash
python music_classification.py
```

3. Model eğitilir, test edilir ve örnek tahmin yapılır. Ardından kullanıcıdan giriş alarak müzik türü tahmini yapılabilir.

## 📊 Örnek Çıktı

```
📊 Model Performansı:

              precision    recall  f1-score   support

     edm         0.85       0.84      0.84       340
     pop         0.86       0.88      0.87       346
     rap         0.83       0.82      0.83       330
    rock         0.87       0.86      0.86       337

    accuracy                         0.85      1353
   macro avg                         0.85
weighted avg                         0.85

🎵 Örnek Şarkının Tahmini Türü: pop (%91.67 güven)
```

## 🧪 Örnek Şarkı Özellikleri

```python
example_song = {
    'danceability': 0.726,
    'energy': 0.816,
    'tempo': 0.9991,
    'loudness': -4.96,
    'valence': 0.693,
    'acousticness': 0.0724,
    'instrumentalness': 0.00421,
    'speechiness': 0.0373,
    'duration_ms': 162600
}
```

## 👨‍💻 Geliştirici

**Selim Bedirhan Öztürk**
Bilgisayar Mühendisliği 3. sınıf öğrencisi
📍 Ankara Üniversitesi
📬 \[[selimbedirhan42@gmail.com]]


