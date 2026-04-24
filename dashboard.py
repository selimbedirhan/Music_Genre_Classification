"""
🎵 Music Genre Classification – Premium Streamlit Dashboard
Run: streamlit run dashboard.py
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.config import (
    DATA_PATH, MODELS_DIR, FIGURES_DIR, RESULTS_PATH,
    BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH,
    GENRE_COLORS, GENRE_EMOJIS, AUDIO_FEATURES,
)
from src.data_pipeline import (
    load_raw_data, clean_data, feature_engineer, ALL_FEATURES
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Genre AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

.metric-card {
    background: linear-gradient(135deg, #1e1e3f, #2d2d5e);
    border: 1px solid #3d3d7a;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px #00d4ff33; }
.metric-value { font-size: 2rem; font-weight: 900; color: #00D4FF; }
.metric-label { font-size: 0.8rem; color: #9090c0; font-weight: 600; letter-spacing: .05em; }

.genre-badge {
    display: inline-block;
    padding: .3rem .9rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    margin: .3rem;
}
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #e0e0f0;
    border-left: 4px solid #00D4FF; padding-left: .8rem; margin: 1.5rem 0 1rem;
}
div[data-testid="stSidebar"] { background: #12122a !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = clean_data(load_raw_data())
    return feature_engineer(df)

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(BEST_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)
        return model, scaler, le
    except Exception:
        return None, None, None

@st.cache_data
def load_results():
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Music Genre AI")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 EDA",
        "🤖 Model Benchmark",
        "🔍 Predict Genre",
        "💡 Explainability",
    ])
    st.markdown("---")
    trained = os.path.exists(BEST_MODEL_PATH)
    if trained:
        st.success("✅ Model Trained")
    else:
        st.warning("⚠️ Run `python train.py` first")


df = load_data()
model, scaler, le = load_artifacts()
results = load_results()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <h1 style='text-align:center; background: linear-gradient(90deg,#00D4FF,#FF6B9D);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:3rem; font-weight:900; margin-bottom:.2rem;'>
    🎵 Music Genre Classification
    </h1>
    <p style='text-align:center; color:#9090c0; font-size:1.1rem; margin-bottom:2rem;'>
    End-to-end ML pipeline · XGBoost · LightGBM · SHAP · 32K+ Spotify tracks
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Tracks", f"{len(df):,}", c1),
        ("Genres", str(df["playlist_genre"].nunique()), c2),
        ("Features", str(len(ALL_FEATURES)), c3),
        ("Best Acc", f"{max((v['accuracy'] for v in results.values()), default=0):.1%}" if results else "—", c4),
    ]
    for label, val, col in metrics:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Genre Distribution</div>", unsafe_allow_html=True)
    counts = df["playlist_genre"].value_counts().reset_index()
    counts.columns = ["genre", "count"]
    counts["emoji"] = counts["genre"].map(GENRE_EMOJIS)
    counts["color"] = counts["genre"].map(GENRE_COLORS)
    fig = px.bar(counts, x="genre", y="count",
                 color="genre", color_discrete_map=GENRE_COLORS,
                 text="count", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                      showlegend=False, height=380)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>🛠 Tech Stack</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for col, tech in zip(cols, ["Python 3.11", "Scikit-learn", "XGBoost", "LightGBM", "SHAP"]):
        col.markdown(f"""<div class='metric-card' style='padding:.7rem'>
        <div style='color:#00D4FF;font-weight:700;font-size:.9rem;'>{tech}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("<h2 class='section-title'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Correlations", "📦 Boxplots"])

    with tab1:
        feat = st.selectbox("Feature", AUDIO_FEATURES)
        fig = px.histogram(df, x=feat, color="playlist_genre",
                           color_discrete_map=GENRE_COLORS,
                           barmode="overlay", nbins=60,
                           template="plotly_dark", opacity=0.7)
        fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e", height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Violin
        fig2 = px.violin(df, x="playlist_genre", y=feat,
                         color="playlist_genre", color_discrete_map=GENRE_COLORS,
                         box=True, template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                           height=420, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        corr = df[AUDIO_FEATURES].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r",
                        template="plotly_dark", title="Feature Correlation Matrix")
        fig.update_layout(paper_bgcolor="#0f0f1a", height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        feat2 = st.selectbox("Feature", AUDIO_FEATURES, key="box_feat")
        fig = px.box(df, x="playlist_genre", y=feat2,
                     color="playlist_genre", color_discrete_map=GENRE_COLORS,
                     template="plotly_dark", points="outliers")
        fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                          height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Radar Chart – Genre Audio Profiles</div>", unsafe_allow_html=True)
    radar_feats = ["danceability", "energy", "valence", "acousticness",
                   "speechiness", "instrumentalness"]
    genre_means = df.groupby("playlist_genre")[radar_feats].mean()
    fig = go.Figure()
    for genre, row in genre_means.iterrows():
        vals = row.tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=radar_feats + [radar_feats[0]],
            fill="toself", name=genre,
            line_color=GENRE_COLORS.get(genre, "#ffffff"),
            opacity=0.7,
        ))
    fig.update_layout(polar=dict(bgcolor="#1a1a2e",
                                  angularaxis=dict(color="#9090c0"),
                                  radialaxis=dict(color="#9090c0")),
                      paper_bgcolor="#0f0f1a", template="plotly_dark",
                      height=500, title="Genre Audio Fingerprints")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Benchmark":
    st.markdown("<h2 class='section-title'>Model Benchmark Results</h2>", unsafe_allow_html=True)

    if not results:
        st.info("Run `python train.py` to generate benchmark results.")
    else:
        names = list(results.keys())
        accs  = [results[n]["accuracy"]    for n in names]
        f1s   = [results[n]["f1_weighted"] for n in names]
        cvs   = [results[n]["cv_mean"]     for n in names]

        fig = go.Figure()
        for vals, label, color in zip(
            [accs, f1s, cvs],
            ["Accuracy", "F1-Weighted", "CV F1"],
            ["#00D4FF", "#FF6B9D", "#FFD700"],
        ):
            fig.add_trace(go.Bar(name=label, x=names, y=vals,
                                  marker_color=color, opacity=0.85))
        fig.update_layout(barmode="group", template="plotly_dark",
                          paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                          height=420, yaxis_range=[0.5, 1.01],
                          title="Accuracy / F1 / CV-F1 per Model")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        rows = []
        for n in names:
            r = results[n]
            rows.append({
                "Model": n,
                "Accuracy": f"{r['accuracy']:.4f}",
                "F1-Weighted": f"{r['f1_weighted']:.4f}",
                "CV F1 (mean)": f"{r['cv_mean']:.4f}",
                "CV F1 (std)": f"±{r['cv_std']:.4f}",
                "Time (s)": r.get("elapsed_s", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Confusion matrix (stored as list)
        st.markdown("<div class='section-title'>Confusion Matrix – Best Model</div>", unsafe_allow_html=True)
        best = max(results, key=lambda k: results[k]["f1_weighted"])
        cm_data = np.array(results[best]["confusion_matrix"])
        if le is not None:
            classes = list(le.classes_)
        else:
            classes = [str(i) for i in range(cm_data.shape[0])]
        fig_cm = px.imshow(cm_data, x=classes, y=classes,
                           text_auto=True, color_continuous_scale="Blues",
                           template="plotly_dark",
                           labels=dict(x="Predicted", y="Actual"))
        fig_cm.update_layout(paper_bgcolor="#0f0f1a", height=500)
        st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT GENRE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Genre":
    st.markdown("<h2 class='section-title'>🎧 Real-Time Genre Prediction</h2>", unsafe_allow_html=True)

    if model is None:
        st.error("Model not found. Run `python train.py` first.")
    else:
        with st.form("predict_form"):
            st.markdown("#### Adjust audio features with the sliders:")
            c1, c2, c3 = st.columns(3)
            with c1:
                danceability     = st.slider("💃 Danceability",     0.0, 1.0, 0.72, 0.01)
                energy           = st.slider("⚡ Energy",           0.0, 1.0, 0.82, 0.01)
                valence          = st.slider("😊 Valence",          0.0, 1.0, 0.69, 0.01)
                liveness         = st.slider("🎤 Liveness",         0.0, 1.0, 0.12, 0.01)
            with c2:
                acousticness     = st.slider("🎸 Acousticness",     0.0, 1.0, 0.07, 0.01)
                instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.00, 0.01)
                speechiness      = st.slider("🗣 Speechiness",      0.0, 1.0, 0.04, 0.01)
                mode             = st.selectbox("🎼 Mode", [0, 1], index=1)
            with c3:
                tempo            = st.slider("🥁 Tempo (BPM)",   50.0, 220.0, 120.0, 1.0)
                loudness         = st.slider("🔊 Loudness (dB)", -40.0, 5.0, -5.0, 0.1)
                duration_ms      = st.slider("⏱ Duration (ms)", 30000, 600000, 200000, 1000)
                key              = st.slider("🎵 Key",           0, 11, 5)
                track_popularity = st.slider("⭐ Popularity",    0, 100, 60)

            submitted = st.form_submit_button("🚀 Predict Genre", use_container_width=True)

        if submitted:
            from src.data_pipeline import feature_engineer
            row = {
                "danceability": danceability, "energy": energy,
                "loudness": loudness, "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness, "valence": valence,
                "tempo": tempo, "duration_ms": duration_ms,
                "key": key, "mode": mode,
                "track_popularity": track_popularity,
            }
            sample_df = pd.DataFrame([row])
            sample_df = feature_engineer(sample_df)
            X_input = sample_df[ALL_FEATURES].values.astype("float32")
            X_scaled = scaler.transform(X_input)

            probs = model.predict_proba(X_scaled)[0]
            pred_idx = int(np.argmax(probs))
            pred_genre = le.classes_[pred_idx]
            confidence = float(probs[pred_idx]) * 100

            color = GENRE_COLORS.get(pred_genre, "#00D4FF")
            emoji = GENRE_EMOJIS.get(pred_genre, "🎵")

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{color}22,{color}44);
            border:2px solid {color}; border-radius:20px; padding:2rem;
            text-align:center; margin:1.5rem 0;'>
              <div style='font-size:4rem;'>{emoji}</div>
              <div style='font-size:2.5rem; font-weight:900; color:{color};'>{pred_genre.upper()}</div>
              <div style='font-size:1.2rem; color:#e0e0f0; margin-top:.5rem;'>
                Confidence: <strong>{confidence:.1f}%</strong>
              </div>
            </div>""", unsafe_allow_html=True)

            # Probability bar chart
            genre_labels = list(le.classes_)
            fig = px.bar(
                x=genre_labels, y=[p * 100 for p in probs],
                color=genre_labels, color_discrete_map=GENRE_COLORS,
                labels={"x": "Genre", "y": "Probability (%)"},
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                              height=350, showlegend=False,
                              title="Prediction Confidence per Genre")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Explainability":
    st.markdown("<h2 class='section-title'>💡 Model Explainability (SHAP)</h2>", unsafe_allow_html=True)

    shap_img = os.path.join(FIGURES_DIR, "shap_summary.png")
    fi_img   = os.path.join(FIGURES_DIR, "feature_importance.png")

    if os.path.exists(shap_img):
        st.image(shap_img, caption="Global SHAP Feature Importance", use_container_width=True)
    else:
        st.info("SHAP plot not found. Run `python train.py` to generate it.")

    if os.path.exists(fi_img):
        st.image(fi_img, caption="Model Feature Importance (built-in)", use_container_width=True)
    else:
        st.info("Feature importance plot not found.")

    # Feature importance from model if available
    if model is not None and hasattr(model, "feature_importances_"):
        st.markdown("<div class='section-title'>Interactive Feature Importance</div>", unsafe_allow_html=True)
        imp = pd.Series(model.feature_importances_, index=ALL_FEATURES)
        imp = imp.sort_values(ascending=True).tail(20)
        fig = px.bar(imp, orientation="h",
                     color=imp.values,
                     color_continuous_scale="plasma",
                     template="plotly_dark",
                     labels={"value": "Importance", "index": "Feature"})
        fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                          height=500, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
