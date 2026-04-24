"""
🎵 Music Genre Classification – Premium Streamlit Dashboard v2
Run: streamlit run dashboard.py
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

from src.config import (
    DATA_PATH, MODELS_DIR, FIGURES_DIR, RESULTS_PATH,
    BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH,
    FEATURE_NAMES_PATH, SHAP_VALUES_PATH,
    GENRE_COLORS, GENRE_EMOJIS, GENRE_DESCRIPTIONS, AUDIO_FEATURES,
)
from src.data_pipeline import load_raw_data, clean_data, feature_engineer, ALL_FEATURES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Genre AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg,#0a0a18 0%,#141428 50%,#0e1a30 100%); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d0d22 0%,#12122e 100%) !important;
    border-right: 1px solid #2a2a5a;
}

.hero-title {
    font-size: 3.6rem; font-weight: 900; text-align: center;
    background: linear-gradient(90deg,#00D4FF,#FF6B9D,#FFD700);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: .4rem;
}
.hero-sub {
    text-align: center; color: #7070a0; font-size: 1.1rem; margin-bottom: 2rem;
}
.kpi-card {
    background: linear-gradient(135deg,#1a1a3e,#252550);
    border: 1px solid #3a3a7a; border-radius: 18px;
    padding: 1.4rem 1.6rem; text-align: center;
    transition: transform .25s, box-shadow .25s;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 10px 30px #00d4ff30; }
.kpi-val  { font-size: 2.2rem; font-weight: 900; color: #00D4FF; }
.kpi-lbl  { font-size: .78rem; color: #8080b0; font-weight: 600; letter-spacing:.06em; text-transform:uppercase; }

.sec-title {
    font-size: 1.35rem; font-weight: 700; color: #e0e0f0;
    border-left: 4px solid #00D4FF; padding-left: .8rem; margin: 1.8rem 0 1rem;
}
.predict-card {
    border-radius: 22px; padding: 2rem; text-align: center; margin: 1.5rem 0;
    transition: transform .3s;
}
.predict-card:hover { transform: scale(1.02); }
.tech-pill {
    display: inline-block; padding: .35rem 1rem;
    border-radius: 20px; font-size: .82rem; font-weight: 700;
    margin: .25rem; border: 1px solid #3a3a7a; color: #a0c4ff;
    background: #1a1a3e;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return feature_engineer(clean_data(load_raw_data()))

@st.cache_resource
def load_artifacts():
    try:
        model  = joblib.load(BEST_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le     = joblib.load(ENCODER_PATH)
        feats  = joblib.load(FEATURE_NAMES_PATH) if os.path.exists(FEATURE_NAMES_PATH) else ALL_FEATURES
        return model, scaler, le, feats
    except Exception:
        return None, None, None, ALL_FEATURES

@st.cache_data
def load_results():
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_shap():
    if os.path.exists(SHAP_VALUES_PATH):
        return joblib.load(SHAP_VALUES_PATH)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <div style='font-size:2.5rem'>🎵</div>
      <div style='font-size:1.1rem;font-weight:900;color:#00D4FF'>Music Genre AI</div>
      <div style='font-size:.75rem;color:#6060a0;margin-top:.2rem'>Professional ML Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠  Overview",
        "📊  EDA",
        "🤖  Benchmark",
        "🎧  Predict",
        "💡  Explainability",
    ], label_visibility="collapsed")

    st.markdown("---")
    trained = os.path.exists(BEST_MODEL_PATH)
    if trained:
        st.success("✅ Model trained & ready")
    else:
        st.warning("⚠️ Run `python train.py` first")

    results_data = load_results()
    if results_data:
        best_k = max(results_data, key=lambda k: results_data[k]["f1_weighted"])
        best_f1 = results_data[best_k]["f1_weighted"]
        best_acc = results_data[best_k]["accuracy"]
        st.metric("Best F1-Score",  f"{best_f1:.2%}")
        st.metric("Best Accuracy",  f"{best_acc:.2%}")
        st.caption(f"🏆 {best_k}")


df = load_data()
model, scaler, le, feat_names = load_artifacts()
results = load_results()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("<div class='hero-title'>🎵 Music Genre Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>End-to-end ML · XGBoost · LightGBM · Stacking Ensemble · SHAP · 32K+ Spotify tracks</div>", unsafe_allow_html=True)

    best_acc = max((v["accuracy"] for v in results.values()), default=0) if results else 0
    best_f1  = max((v["f1_weighted"] for v in results.values()), default=0) if results else 0
    kpis = [
        ("Total Tracks",  f"{len(df):,}"),
        ("Genres",        str(df["playlist_genre"].nunique())),
        ("Features",      str(len(feat_names))),
        ("Best Accuracy", f"{best_acc:.1%}" if results else "—"),
        ("Best F1",       f"{best_f1:.1%}"  if results else "—"),
    ]
    cols = st.columns(5)
    for col, (lbl, val) in zip(cols, kpis):
        col.markdown(f"""<div class='kpi-card'>
          <div class='kpi-val'>{val}</div>
          <div class='kpi-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>Genre Distribution</div>", unsafe_allow_html=True)
    counts = df["playlist_genre"].value_counts().reset_index()
    counts.columns = ["genre", "count"]
    fig = px.bar(counts, x="genre", y="count", color="genre",
                 color_discrete_map=GENRE_COLORS, text="count",
                 template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0a0a18", plot_bgcolor="#141428",
                      showlegend=False, height=360,
                      margin=dict(l=20, r=20, t=20, b=20))
    fig.update_traces(textposition="outside",
                      marker=dict(line=dict(color="rgba(255,255,255,0.1)", width=0.5)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-title'>Genre DNA – Audio Fingerprints</div>", unsafe_allow_html=True)
    radar_feats = ["danceability", "energy", "valence", "acousticness", "speechiness", "liveness"]
    genre_means = df.groupby("playlist_genre")[radar_feats].mean()
    fig2 = go.Figure()
    for genre, row in genre_means.iterrows():
        v = row.tolist()
        fig2.add_trace(go.Scatterpolar(
            r=v + [v[0]], theta=radar_feats + [radar_feats[0]],
            fill="toself", name=f"{GENRE_EMOJIS.get(genre,'')} {genre}",
            line_color=GENRE_COLORS.get(genre, "#ffffff"), opacity=0.75,
        ))
    fig2.update_layout(
        polar=dict(bgcolor="#141428",
                   angularaxis=dict(color="#7070a0"),
                   radialaxis=dict(color="#7070a0", gridcolor="#2a2a4a")),
        paper_bgcolor="#0a0a18", template="plotly_dark",
        height=480, legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='sec-title'>🛠 Tech Stack</div>", unsafe_allow_html=True)
    techs = ["Python 3.11", "Scikit-learn", "XGBoost", "LightGBM",
             "Stacking Ensemble", "Optuna HPO", "SHAP", "Streamlit", "Plotly"]
    st.markdown(" ".join(f"<span class='tech-pill'>{t}</span>" for t in techs),
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown("<div class='sec-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔥 Correlations", "📦 Boxplots", "🎻 Violins"])

    BG1, BG2 = "#0a0a18", "#141428"

    with tab1:
        feat = st.selectbox("Select feature", AUDIO_FEATURES, key="hist_feat")
        fig = px.histogram(df, x=feat, color="playlist_genre",
                           color_discrete_map=GENRE_COLORS,
                           barmode="overlay", nbins=60,
                           template="plotly_dark", opacity=0.72)
        fig.update_layout(paper_bgcolor=BG1, plot_bgcolor=BG2, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = df[AUDIO_FEATURES].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r",
                        template="plotly_dark", title="Feature Correlation Matrix")
        fig.update_layout(paper_bgcolor=BG1, height=560)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        feat3 = st.selectbox("Select feature", AUDIO_FEATURES, key="box_feat")
        fig = px.box(df, x="playlist_genre", y=feat3,
                     color="playlist_genre", color_discrete_map=GENRE_COLORS,
                     template="plotly_dark", points="outliers")
        fig.update_layout(paper_bgcolor=BG1, plot_bgcolor=BG2,
                          height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        feat4 = st.selectbox("Select feature", AUDIO_FEATURES, key="viol_feat")
        fig = px.violin(df, x="playlist_genre", y=feat4,
                        color="playlist_genre", color_discrete_map=GENRE_COLORS,
                        box=True, template="plotly_dark")
        fig.update_layout(paper_bgcolor=BG1, plot_bgcolor=BG2,
                          height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Benchmark":
    st.markdown("<div class='sec-title'>Model Benchmark Results</div>", unsafe_allow_html=True)

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
                                  marker_color=color, opacity=0.88,
                                  marker_line=dict(color="#ffffff11", width=0.5)))
        fig.update_layout(barmode="group", template="plotly_dark",
                          paper_bgcolor="#0a0a18", plot_bgcolor="#141428",
                          height=420, yaxis_range=[0.4, 1.01],
                          title="Accuracy / F1 / CV-F1 per Model",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        rows = []
        best_f1_val = max(r["f1_weighted"] for r in results.values())
        for n in names:
            r = results[n]
            rows.append({
                "Model": ("🏆 " if r["f1_weighted"] == best_f1_val else "") + n,
                "Accuracy":     f"{r['accuracy']:.4f}",
                "F1-Weighted":  f"{r['f1_weighted']:.4f}",
                "CV F1 (mean)": f"{r['cv_mean']:.4f}",
                "CV F1 (std)":  f"±{r['cv_std']:.4f}",
                "Time (s)":     r.get("elapsed_s", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Confusion matrix
        st.markdown("<div class='sec-title'>Confusion Matrix – Best Model</div>", unsafe_allow_html=True)
        best_k  = max(results, key=lambda k: results[k]["f1_weighted"])
        cm_data = np.array(results[best_k]["confusion_matrix"])
        classes = list(le.classes_) if le else [str(i) for i in range(cm_data.shape[0])]

        fig_cm = make_subplots(rows=1, cols=2, subplot_titles=["Counts", "Normalised"])
        cm_norm = cm_data.astype(float) / cm_data.sum(axis=1, keepdims=True)
        for col_i, (data, fmt) in enumerate([(cm_data, "d"), (cm_norm, ".2f")], 1):
            fig_cm.add_trace(go.Heatmap(
                z=data, x=classes, y=classes,
                colorscale="Blues", showscale=False,
                text=[[f"{v:{fmt}}" for v in row] for row in data],
                texttemplate="%{text}",
            ), row=1, col=col_i)
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="#0a0a18",
                             height=460)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class report
        st.markdown("<div class='sec-title'>Per-Class Metrics – Best Model</div>", unsafe_allow_html=True)
        report = results[best_k].get("report", {})
        report_rows = []
        for cls in classes:
            if cls in report:
                m = report[cls]
                report_rows.append({
                    "Genre":     f"{GENRE_EMOJIS.get(cls,'')} {cls}",
                    "Precision": f"{m['precision']:.4f}",
                    "Recall":    f"{m['recall']:.4f}",
                    "F1-Score":  f"{m['f1-score']:.4f}",
                    "Support":   int(m["support"]),
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎧  Predict":
    st.markdown("<div class='sec-title'>🎧 Real-Time Genre Prediction</div>", unsafe_allow_html=True)

    if model is None:
        st.error("Model not found. Run `python train.py` first.")
    else:
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🎼 Rhythm & Energy**")
                danceability = st.slider("💃 Danceability",   0.0, 1.0, 0.72, 0.01)
                energy       = st.slider("⚡ Energy",         0.0, 1.0, 0.82, 0.01)
                tempo        = st.slider("🥁 Tempo (BPM)",   50.0, 220.0, 120.0, 1.0)
                valence      = st.slider("😊 Valence",        0.0, 1.0, 0.69, 0.01)
            with c2:
                st.markdown("**🎸 Texture**")
                acousticness     = st.slider("🎸 Acousticness",     0.0, 1.0, 0.07, 0.01)
                instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.00, 0.01)
                speechiness      = st.slider("🗣 Speechiness",      0.0, 1.0, 0.04, 0.01)
                liveness         = st.slider("🎤 Liveness",         0.0, 1.0, 0.12, 0.01)
            with c3:
                st.markdown("**🔊 Production**")
                loudness         = st.slider("🔊 Loudness (dB)",  -40.0,  5.0,  -5.0, 0.1)
                duration_ms      = st.slider("⏱ Duration (ms)", 30000, 600000, 200000, 1000)
                key              = st.slider("🎵 Key",            0, 11, 5)
                mode             = st.selectbox("🎼 Mode", [0, 1],
                                                format_func=lambda x: "Major" if x else "Minor")
                track_popularity = st.slider("⭐ Popularity", 0, 100, 60)

            submitted = st.form_submit_button("🚀 Predict Genre", use_container_width=True)

        if submitted:
            from src.data_pipeline import feature_engineer as fe
            row = {
                "danceability": danceability, "energy": energy,
                "loudness": loudness, "speechiness": speechiness,
                "acousticness": acousticness, "instrumentalness": instrumentalness,
                "liveness": liveness, "valence": valence, "tempo": tempo,
                "duration_ms": duration_ms, "key": key, "mode": mode,
                "track_popularity": track_popularity,
            }
            sample_df  = fe(pd.DataFrame([row]))
            X_input    = sample_df[feat_names].values.astype("float32")
            X_scaled   = scaler.transform(X_input)
            probs      = model.predict_proba(X_scaled)[0]
            pred_idx   = int(np.argmax(probs))
            pred_genre = le.classes_[pred_idx]
            confidence = float(probs[pred_idx]) * 100
            color      = GENRE_COLORS.get(pred_genre, "#00D4FF")
            emoji      = GENRE_EMOJIS.get(pred_genre, "🎵")
            desc       = GENRE_DESCRIPTIONS.get(pred_genre, "")

            st.markdown(f"""
            <div class='predict-card'
                 style='background:linear-gradient(135deg,{color}18,{color}35);
                        border:2px solid {color};'>
              <div style='font-size:5rem'>{emoji}</div>
              <div style='font-size:3rem;font-weight:900;color:{color}'>{pred_genre.upper()}</div>
              <div style='color:#c0c0e0;margin:.5rem 0'>{desc}</div>
              <div style='font-size:1.4rem;color:#e0e0f0'>
                Confidence: <strong style='color:{color}'>{confidence:.1f}%</strong>
              </div>
            </div>""", unsafe_allow_html=True)

            genre_labels = list(le.classes_)
            sorted_pairs = sorted(zip(genre_labels, probs), key=lambda x: x[1], reverse=True)
            fig = go.Figure(go.Bar(
                x=[p * 100 for _, p in sorted_pairs],
                y=[g for g, _ in sorted_pairs],
                orientation="h",
                marker=dict(
                    color=[GENRE_COLORS.get(g, "#8080aa") for g, _ in sorted_pairs],
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                ),
                text=[f"{p*100:.1f}%" for _, p in sorted_pairs],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0a18",
                plot_bgcolor="#141428", height=320, showlegend=False,
                xaxis_title="Probability (%)",
                title="Genre Prediction Confidence",
                margin=dict(l=80, r=60, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡  Explainability":
    st.markdown("<div class='sec-title'>💡 Model Explainability (SHAP)</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌐 Global SHAP", "🐝 Beeswarm", "📊 Feature Importance"])

    with tab1:
        shap_img = os.path.join(FIGURES_DIR, "shap_summary.png")
        if os.path.exists(shap_img):
            st.image(shap_img, caption="Global SHAP – Mean |SHAP| across all classes",
                     use_container_width=True)
        else:
            st.info("Run `python train.py` to generate SHAP plots.")

        # Interactive SHAP bar from cached values
        shap_cache = load_shap()
        if shap_cache and model is not None and hasattr(model, "feature_importances_"):
            sv   = shap_cache["shap_values"]
            fns  = shap_cache.get("feature_names", feat_names)
            if isinstance(sv, list):
                abs_mean = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
            else:
                abs_mean = np.abs(sv).mean(axis=0)
            importance_df = pd.DataFrame({"feature": fns, "shap": abs_mean})
            importance_df = importance_df.nlargest(20, "shap").sort_values("shap")
            fig = px.bar(importance_df, x="shap", y="feature", orientation="h",
                         color="shap", color_continuous_scale="plasma",
                         template="plotly_dark",
                         labels={"shap": "Mean |SHAP|", "feature": "Feature"})
            fig.update_layout(paper_bgcolor="#0a0a18", plot_bgcolor="#141428",
                              height=540, coloraxis_showscale=False,
                              title="Interactive Global SHAP Importance")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        class_names = list(le.classes_) if le else []
        if class_names:
            selected_genre = st.selectbox("Select genre for beeswarm", class_names)
            idx = class_names.index(selected_genre)
            bee_img = os.path.join(FIGURES_DIR, f"shap_beeswarm_class{idx}.png")
            if os.path.exists(bee_img):
                st.image(bee_img, caption=f"SHAP Beeswarm – {selected_genre}",
                         use_container_width=True)
            else:
                st.info("Beeswarm plots not found. Run `python train.py`.")
        else:
            st.info("Load model artifacts first.")

    with tab3:
        fi_img = os.path.join(FIGURES_DIR, "feature_importance.png")
        if os.path.exists(fi_img):
            st.image(fi_img, caption="Built-in Feature Importance", use_container_width=True)

        if model is not None and hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feat_names)
            imp = imp.sort_values(ascending=True).tail(20)
            fig = px.bar(imp, orientation="h", color=imp.values,
                         color_continuous_scale="viridis",
                         template="plotly_dark",
                         labels={"value": "Importance", "index": "Feature"})
            fig.update_layout(paper_bgcolor="#0a0a18", plot_bgcolor="#141428",
                              height=520, coloraxis_showscale=False,
                              title="Interactive Feature Importance (Top 20)")
            st.plotly_chart(fig, use_container_width=True)
        elif model is not None:
            st.info("Selected model is a Stacking Ensemble — use Global SHAP tab for importance.")
