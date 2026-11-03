import io
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import utils  # noqa: F401  (nécessaire pour recharger IQRClipper depuis les .pkl)

st.set_page_config(page_title="ML Predictor", page_icon="🔮", layout="wide")

MODELS_DIR = Path(__file__).parent / "models"
MODELS = {
    "telephone": {"icon": "📱", "label": "Téléphones : état", "unit": "",
                  "desc": "Prédit l'état d'un téléphone (Neuf, Réconditionné, Venant, D'occasion)."},
    "voiture_etat": {"icon": "🚗", "label": "Voitures : Venant / D'occasion", "unit": "",
                     "desc": "Prédit si une voiture est « Venant » ou « D'occasion »."},
    "voiture_prix": {"icon": "💰", "label": "Voitures : prix de vente", "unit": "",
                     "desc": "Estime le prix de vente d'une voiture (unité du dataset Car_dataset)."},
    "taxi_prix": {"icon": "🚕", "label": "Taxi : prix de la course", "unit": "",
                  "desc": "Estime le prix d'une course de taxi (unité du dataset taxi_trip_pricing)."},
}

# ---------- Style (thème sombre, voir aussi .streamlit/config.toml) ----------
st.markdown("""
<style>
.hero{padding:1.4rem 1.6rem;border-radius:18px;margin-bottom:1rem;
  background:linear-gradient(135deg,rgba(124,92,255,.35),rgba(0,212,255,.18));
  border:1px solid rgba(255,255,255,.12)}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.3rem 0 0;opacity:.8}
.card{padding:1.2rem 1.4rem;border-radius:16px;border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.04);backdrop-filter:blur(6px);margin:.5rem 0}
.big{font-size:2.4rem;font-weight:800;line-height:1.1;
  background:linear-gradient(90deg,#7c5cff,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.small{opacity:.7;font-size:.9rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle(key):
    return joblib.load(MODELS_DIR / f"{key}.pkl")


def num_widget(col, info, key):
    """Champ numérique adapté à la colonne (entier / décimal, pas cohérent)."""
    lo, hi, med = info["min"], info["max"], info["median"]
    if info["is_int"]:
        step = 1000 if med >= 100_000 else 1
        return st.number_input(col, min_value=int(min(lo, 0)) if lo >= 0 else int(lo),
                               value=int(med), step=step, key=key)
    step = 0.01 if hi < 10 else (0.1 if hi < 200 else 1.0)
    return st.number_input(col, min_value=float(min(lo, 0.0)) if lo >= 0 else float(lo),
                           value=float(round(med, 2)), step=step, key=key)


def predict_df(bundle, df):
    """Retourne un DataFrame de prédictions (+ confiance pour la classification)."""
    X = df[bundle["num_cols"] + bundle["cat_cols"]].copy()
    for c in bundle["num_cols"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    pipe = bundle["pipeline"]
    out = pd.DataFrame(index=df.index)
    if bundle["task"] == "classification":
        proba = pipe.predict_proba(X)
        out[f"{bundle['target']} (prédit)"] = pipe.classes_[proba.argmax(axis=1)]
        out["Confiance (%)"] = (proba.max(axis=1) * 100).round(1)
        return out, pd.DataFrame(proba, columns=pipe.classes_, index=df.index)
    out[f"{bundle['target']} (prédit)"] = pipe.predict(X).round(2)
    return out, None


def template_csv(bundle):
    row = {c: bundle["num_info"][c]["median"] for c in bundle["num_cols"]}
    for c in bundle["cat_cols"]:
        row[c] = bundle["cat_values"][c][0]
    for c, v in row.items():
        if c in bundle["num_info"] and bundle["num_info"][c]["is_int"]:
            row[c] = int(v)
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig")


def model_header(bundle):
    m = bundle["test_metrics"]
    cols = st.columns(len(m) + 1)
    cols[0].metric("Algorithme retenu", bundle["best_name"])
    for c, (k, v) in zip(cols[1:], m.items()):
        c.metric(f"{k} (test)", f"{v:.3f}")


# ---------- Onglet 1 : formulaire ----------
def tab_form(key, bundle):
    st.caption("Remplis les informations puis clique sur **Prédire**.")
    with st.form(f"form_{key}"):
        values, cols = {}, st.columns(2)
        fields = bundle["num_cols"] + bundle["cat_cols"]
        for i, col in enumerate(fields):
            with cols[i % 2]:
                if col in bundle["num_cols"]:
                    values[col] = num_widget(col, bundle["num_info"][col], f"{key}_{col}")
                else:
                    values[col] = st.selectbox(col, bundle["cat_values"][col], key=f"{key}_{col}")
        submitted = st.form_submit_button("🔮 Prédire", width="stretch", type="primary")
    if not submitted:
        return
    pred, proba = predict_df(bundle, pd.DataFrame([values]))
    value = pred.iloc[0, 0]
    if bundle["task"] == "classification":
        st.markdown(f'<div class="card"><div class="small">Prédiction</div>'
                    f'<div class="big">{value}</div>'
                    f'<div class="small">Confiance : {pred["Confiance (%)"].iloc[0]} %</div></div>',
                    unsafe_allow_html=True)
        p = proba.iloc[0].sort_values().reset_index()
        p.columns = ["Classe", "Probabilité"]
        fig = px.bar(p, x="Probabilité", y="Classe", orientation="h", text_auto=".1%",
                     color="Probabilité", color_continuous_scale=["#3b2f8f", "#7c5cff", "#00d4ff"])
        fig.update_layout(height=260, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    else:
        st.markdown(f'<div class="card"><div class="small">{bundle["target"]} estimé</div>'
                    f'<div class="big">{value:,.2f}</div></div>', unsafe_allow_html=True)


# ---------- Onglet 2 : fichier CSV ----------
def tab_csv(key, bundle):
    needed = bundle["num_cols"] + bundle["cat_cols"]
    st.caption("Le fichier CSV doit contenir ces colonnes (les autres colonnes sont conservées/ignorées) :")
    st.code(", ".join(needed))
    st.download_button("⬇️ Télécharger un modèle de CSV", template_csv(bundle),
                       file_name=f"modele_{key}.csv", mime="text/csv", key=f"tpl_{key}")
    file = st.file_uploader("Importer un fichier CSV", type=["csv"], key=f"up_{key}")
    if file is None:
        return
    try:
        df = pd.read_csv(file, sep=None, engine="python")  # détecte , ou ;
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {', '.join(missing)}")
        return
    with st.spinner("Calcul des prédictions..."):
        pred, _ = predict_df(bundle, df)
    result = pd.concat([df, pred], axis=1)
    st.success(f"{len(result)} ligne(s) traitée(s).")
    st.dataframe(result, width="stretch", height=420)
    st.download_button("⬇️ Télécharger les résultats (CSV)",
                       result.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"predictions_{key}.csv", mime="text/csv", key=f"dl_{key}")
    target = pred.columns[0]
    if bundle["task"] == "classification":
        vc = result[target].value_counts().reset_index()
        vc.columns = ["Classe", "Nombre"]
        fig = px.bar(vc, x="Classe", y="Nombre", text_auto=True, color="Classe")
    else:
        fig = px.histogram(result, x=target, nbins=30)
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig, width="stretch")


# ---------- Pages ----------
def page_home():
    st.markdown('<div class="hero"><h1>🔮 ML Predictor</h1>'
                '<p>4 modèles de machine learning : 2 classifications et 2 régressions. '
                'Choisis un modèle dans le menu de gauche.</p></div>', unsafe_allow_html=True)
    rows = []
    for key, meta in MODELS.items():
        b = load_bundle(key)
        main = list(b["test_metrics"].items())[-1 if b["task"] == "classification" else 0]
        rows.append({"": meta["icon"], "Modèle": meta["label"],
                     "Type": "Classification" if b["task"] == "classification" else "Régression",
                     "Cible": b["target"], "Algorithme": b["best_name"],
                     "Score (test)": f"{main[0]} = {main[1]:.3f}", "Lignes": b["n_rows"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.info("Chaque modèle propose un **formulaire** (prédiction unique) et un **import CSV** "
            "(prédiction pour chaque ligne du fichier).")


def page_model(key):
    meta, bundle = MODELS[key], load_bundle(key)
    st.markdown(f'<div class="hero"><h1>{meta["icon"]} {meta["label"]}</h1><p>{meta["desc"]}</p></div>',
                unsafe_allow_html=True)
    model_header(bundle)
    t1, t2 = st.tabs(["📝 Formulaire", "📂 Fichier CSV"])
    with t1:
        tab_form(key, bundle)
    with t2:
        tab_csv(key, bundle)


with st.sidebar:
    st.title("🔮 ML Predictor")
    choice = st.radio("Navigation", ["🏠 Accueil"] + [f"{m['icon']} {m['label']}" for m in MODELS.values()],
                      label_visibility="collapsed")

if choice.startswith("🏠"):
    page_home()
else:
    page_model(list(MODELS)[[f"{m['icon']} {m['label']}" for m in MODELS.values()].index(choice)])
