import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

csv_path = 'archive/spotify_songs.csv'
df = pd.read_csv(csv_path)

selected_features = [
    'danceability', 'energy', 'tempo', 'loudness', 'valence',
    'acousticness', 'instrumentalness', 'speechiness', 'duration_ms'
]

df = df.dropna(subset=selected_features + ['playlist_genre'])

X = df[selected_features]
y = df['playlist_genre']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("📊 Model Performansı:\n")
print(classification_report(y_test, y_pred))

def predict_genre(features_dict):
    features_array = np.array([[
        features_dict['danceability'],
        features_dict['energy'],
        features_dict['tempo'],
        features_dict['loudness'],
        features_dict['valence'],
        features_dict['acousticness'],
        features_dict['instrumentalness'],
        features_dict['speechiness'],
        features_dict['duration_ms']
    ]])
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled).max() * 100
    return prediction, confidence

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

genre, confidence = predict_genre(example_song)
print(f"\n🎵 Örnek Şarkının Tahmini Türü: {genre} (%{confidence:.2f} güven)")

def get_user_input():
    print("\n🧑‍🎤 Müzik Özelliklerini Girin:")
    try:
        return {
            'danceability': float(input("Danceability (0-1): ")),
            'energy': float(input("Energy (0-1): ")),
            'tempo': float(input("Tempo (BPM): ")),
            'loudness': float(input("Loudness (örnek: -7): ")),
            'valence': float(input("Valence (0-1): ")),
            'acousticness': float(input("Acousticness (0-1): ")),
            'instrumentalness': float(input("Instrumentalness (0-1): ")),
            'speechiness': float(input("Speechiness (0-1): ")),
            'duration_ms': float(input("Süre (ms cinsinden): "))
        }
    except Exception as e:
        print(f"Hata: {e}")
        return None

if input("\n🎧 Kendi şarkını tahmin etmek ister misin? (E/H): ").strip().upper() == 'E':
    user_features = get_user_input()
    if user_features:
        predicted_genre, conf = predict_genre(user_features)
        print(f"\n🎶 Tahmin Edilen Tür: {predicted_genre} (%{conf:.2f} güven)")
else:
    print("Çıkılıyor... 🎤")
