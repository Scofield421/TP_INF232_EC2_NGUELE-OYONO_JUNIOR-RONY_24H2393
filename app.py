import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from supabase import create_client

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ----------------------------
# Supabase client (ANON)
# ----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ----------------------------
# Helpers
# ----------------------------
def validate_range(x, lo, hi):
    if x is None:
        return False
    v = int(x)
    return lo <= v <= hi

def load_data():
    res = (
        supabase.table("ia_africa_responses")
        .select("*")
        .order("created_at", desc=False)
        .execute()
    )
    rows = res.data or []
    return pd.DataFrame(rows)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # Target pour classification supervisée (Q17)
    df["high_adoption"] = (df["adoption_intent"].astype(float) >= 4).astype(int)

    # Colonnes numériques (conversion robuste)
    num_cols = [
        "age",
        "ai_knowledge",
        "internet_access",
        "usage_freq",
        "impact_jobs",
        "impact_education",
        "impact_economy",
        "trust_ai",
        "bias_risk",
        "privacy_risk",
        "misinfo_risk",
        "adoption_intent",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # created_at pour les courbes
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    return df

def safe_dropna(df, cols):
    df2 = df.copy()
    before = len(df2)
    df2 = df2.dropna(subset=cols)
    after = len(df2)
    return df2, before, after

def storytelling_parts(df, reg_simple, reg_multi, clf_results, clusters):
    parts = []
    if df.empty or len(df) < 5:
        return ["Pas assez de données pour produire un storytelling fiable. Ajoute plus de réponses."]

    coef_simple = reg_simple.get("coef_ai_knowledge", None)
    if coef_simple is not None:
        direction = "augmente" if coef_simple > 0 else "diminue"
        parts.append(
            f"**Régression simple (Q6 → Q8) :** quand la **connaissance de l’IA** augmente, la **fréquence d’utilisation** {direction} "
            f"(coefficient ≈ {coef_simple:.3f})."
        )

    top_predictors = reg_multi.get("top_predictors", [])
    if top_predictors:
        bullets = "\n".join([f"- {name} : {val:+.3f}" for name, val in top_predictors[:5]])
        parts.append(
            "**Régression multiple (Q8) :** principaux facteurs associés à la fréquence d’utilisation :\n"
            f"{bullets}"
        )

    if clf_results.get("model_name") is not None:
        acc = clf_results.get("accuracy")
        parts.append(
            f"**Classification supervisée (adoption élevée) :** le meilleur modèle (**{clf_results['model_name']}**) obtient une accuracy "
            f"≈ **{acc:.2%}**."
        )

    if clusters.get("cluster_profiles"):
        parts.append("**Clustering non supervisé (profils) :**")
        for name, desc in clusters["cluster_profiles"].items():
            parts.append(f"- **{name}** : {desc}")

    parts.append(
        "### Recommandations (à partir des tendances)\n"
        "- Renforcer la **formation** et l’**accès** à l’IA (liés à l’usage et à l’adoption).\n"
        "- Agir sur la **confiance** et réduire les **risques perçus** (biais, confidentialité, désinformation).\n"
        "- Adapter les actions aux **profils** ressortis par le clustering."
    )
    return parts


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="IA Afrique - Collecte & Analyse", layout="wide")
st.title("IA Afrique : collecte anonyme & analyse descriptive (ML)")

nav = st.sidebar.radio(
    "Navigation",
    ["Collecte", "Données collectées", "Analyse", "Storytelling"],
    index=0,
)

df_raw = load_data()
df = add_features(df_raw)

# ----------------------------
# COLLECTE (17 questions)
# ----------------------------
if nav == "Collecte":
    st.subheader("Formulaire de collecte (anonyme)")
    st.caption("Le formulaire n’a pas de limite de temps côté application : le répondant remplit puis clique sur “Envoyer”.")

    with st.form("collect_form", clear_on_submit=True):
        st.markdown("## A. Profil (Q1–Q5)")
        country = st.selectbox(
            "Q1) Pays",
            ["Sénégal", "Côte d'Ivoire", "Cameroun", "Maroc", "Kenya", "Nigeria", "Ghana", "Autre"],
        )
        age = st.number_input("Q2) Âge", min_value=15, max_value=90, value=25, step=1)
        education_level = st.selectbox(
            "Q3) Niveau d’études",
            ["Secondaire", "Licence", "Master", "Doctorat", "Autodidacte", "Autre"],
        )
        sector = st.selectbox(
            "Q4) Domaine/secteur principal",
            ["Éducation", "Santé", "Finance", "Administration", "Agriculture", "Commerce/Services", "Technologie", "Autre"],
        )
        training_label = st.selectbox(
            "Q5) Avez-vous déjà suivi une formation/atelier sur l’IA ?",
            ["Non", "Oui"],
        )
        training = True if training_label == "Oui" else False

        st.markdown("---")
        st.markdown("## B. Connaissance & accès (Q6–Q7)")
        ai_knowledge = st.slider("Q6) Connaissance de l’IA (1–5)", 1, 5, 3)
        internet_access = st.slider("Q7) Accès aux outils/internet pour utiliser l’IA (1–5)", 1, 5, 3)

        st.markdown("---")
        st.markdown("## C. Utilisation (Q8–Q9)")
        usage_freq = st.selectbox(
            "Q8) Fréquence d’utilisation de l’IA",
            [0, 1, 2, 3, 4],
            format_func=lambda v: {
                0: "0 - Jamais",
                1: "1 - Rarement",
                2: "2 - Mensuellement",
                3: "3 - Hebdomadaire",
                4: "4 - Quotidiennement",
            }[v],
        )
        primary_use_case = st.selectbox(
            "Q9) Objectif principal d’utilisation",
            [
                "Recherche académique",
                "Travail (automatisation)",
                "Création de contenu",
                "Support client",
                "Santé",
                "Finance",
                "Éducation/formation",
                "Autre",
            ],
        )

        st.markdown("---")
        st.markdown("## D. Impact perçu (Q10–Q12) - échelle 1 à 5")
        impact_jobs = st.slider("Q10) Impact sur l’emploi (1–5)", 1, 5, 3)
        impact_education = st.slider("Q11) Impact sur l’éducation (1–5)", 1, 5, 3)
        impact_economy = st.slider("Q12) Impact sur l’économie / productivité (1–5)", 1, 5, 3)

        st.markdown("---")
        st.markdown("## E. Confiance & risques (Q13–Q16) - échelle 1 à 5")
        trust_ai = st.slider("Q13) Confiance dans l’IA (1–5)", 1, 5, 3)
        bias_risk = st.slider("Q14) Crainte de biais / discrimination (1–5)", 1, 5, 3)
        privacy_risk = st.slider("Q15) Crainte pour la confidentialité / données personnelles (1–5)", 1, 5, 3)
        misinfo_risk = st.slider("Q16) Crainte de désinformation / erreurs (1–5)", 1, 5, 3)

        st.markdown("---")
        st.markdown("## F. Adoption ( échelle 1 à 5")
        adoption_intent = st.slider(
            "Q17) Intention d’adopter l’IA dans les 6 prochains mois (1–5)",
            1, 5, 3,
        )

        submitted = st.form_submit_button("Envoyer ma réponse")

    if submitted:
        checks = [
            (ai_knowledge, 1, 5),
            (internet_access, 1, 5),
            (usage_freq, 0, 4),
            (impact_jobs, 1, 5),
            (impact_education, 1, 5),
            (impact_economy, 1, 5),
            (trust_ai, 1, 5),
            (bias_risk, 1, 5),
            (privacy_risk, 1, 5),
            (misinfo_risk, 1, 5),
            (adoption_intent, 1, 5),
        ]
        ok = all(validate_range(v, lo, hi) for v, lo, hi in checks)

        if not ok:
            st.error("Certaines valeurs sont hors plage. Vérifie tes réponses.")
        else:
            payload = {
                "country": country,
                "age": int(age),
                "education_level": education_level,
                "sector": sector,
                "training": bool(training),

                "ai_knowledge": int(ai_knowledge),
                "internet_access": int(internet_access),

                "usage_freq": int(usage_freq),
                "primary_use_case": primary_use_case,

                "impact_jobs": int(impact_jobs),
                "impact_education": int(impact_education),
                "impact_economy": int(impact_economy),

                "trust_ai": int(trust_ai),
                "bias_risk": int(bias_risk),
                "privacy_risk": int(privacy_risk),
                "misinfo_risk": int(misinfo_risk),

                "adoption_intent": int(adoption_intent),
            }

            try:
                supabase.table("ia_africa_responses").insert(payload).execute()
                st.success("Merci ! Ta réponse a été enregistrée.")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de l’insertion Supabase : {e}")

# ----------------------------
# DONNÉES
# ----------------------------
elif nav == "Données collectées":
    st.subheader("Données collectées (anonymes)")
    if df_raw.empty:
        st.warning("Aucune donnée pour le moment. Commence par aller sur ‘Collecte’.")
    else:
        st.write(f"Nombre de réponses : **{len(df_raw)}**")
        st.dataframe(df_raw, use_container_width=True)

        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger CSV", csv, "ia_africa_responses.csv", "text/csv")

# ----------------------------
# ANALYSE
# ----------------------------
elif nav == "Analyse":
    st.subheader("Analyse descriptive + modèles ML")

    if df.empty or len(df) < 5:
        st.warning("Pas assez de données pour lancer l’analyse. Ajoute au moins quelques réponses.")
        st.stop()

    st.write(f"Nombre de réponses : **{len(df)}**")

    # (6) Histogrammes + (8) Camembert + (7) Courbes
    st.markdown("## Histogrammes")
    col1, col2 = st.columns(2)
    with col1:
        fig_h1 = px.histogram(df, x="usage_freq", nbins=5, title="Distribution de la fréquence d’utilisation (Q8)")
        st.plotly_chart(fig_h1, use_container_width=True)
    with col2:
        fig_h2 = px.histogram(df, x="trust_ai", nbins=5, title="Distribution de la confiance dans l’IA (Q13)")
        st.plotly_chart(fig_h2, use_container_width=True)

    st.markdown("## Diagramme de camembert")
    fig_pie = px.pie(df, names="primary_use_case", title="Répartition des objectifs principaux (Q9)")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("## Graphes de courbe")
    df_time = df.dropna(subset=["created_at"]).copy()
    if df_time.empty:
        st.info("created_at indisponible pour tracer des courbes.")
    else:
        df_time["week"] = df_time["created_at"].dt.to_period("W").dt.start_time
        agg = df_time.groupby("week").agg(
            n=("id", "count"),
            usage_mean=("usage_freq", "mean"),
            adoption_mean=("adoption_intent", "mean"),
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            fig_line1 = px.line(agg, x="week", y="usage_mean", markers=True, title="Moyenne de l’utilisation (Q8) par semaine")
            st.plotly_chart(fig_line1, use_container_width=True)
        with c2:
            fig_line2 = px.line(agg, x="week", y="adoption_mean", markers=True, title="Moyenne de l’intention d’adoption (Q17) par semaine")
            st.plotly_chart(fig_line2, use_container_width=True)

    # (1) Régression linéaire simple : usage_freq ~ ai_knowledge
    st.markdown("## Régression linéaire simple : usage_freq ~ ai_knowledge (Q8 ~ Q6)")
    df_reg1, _, _ = safe_dropna(df, ["usage_freq", "ai_knowledge"])
    if len(df_reg1) < 3:
        st.warning("Pas assez de données pour la régression simple.")
    else:
        X = df_reg1[["ai_knowledge"]].values
        y = df_reg1["usage_freq"].values
        model1 = LinearRegression().fit(X, y)

        y_pred = model1.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        coef = float(model1.coef_[0])
        intercept = float(model1.intercept_)

        st.write(f"Équation : **usage_freq = {intercept:.3f} + {coef:.3f} * ai_knowledge**")
        st.write(f"R² = **{r2:.3f}** | MSE = **{mse:.3f}**")

        fig_scatter = px.scatter(df_reg1, x="ai_knowledge", y="usage_freq", opacity=0.75, title="Nuage de points : Q6 vs Q8")
        x_line = np.linspace(df_reg1["ai_knowledge"].min(), df_reg1["ai_knowledge"].max(), 100)
        y_line = intercept + coef * x_line
        fig_scatter.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Régression", line=dict(color="red")))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # (2) Régression linéaire multiple
    st.markdown("## Régression linéaire multiple : usage_freq ~ (Q6,Q7,Q5,Q13,Q12)")
    df_reg2, _, _ = safe_dropna(
        df, ["usage_freq", "ai_knowledge", "internet_access", "training", "trust_ai", "impact_economy"]
    )
    if len(df_reg2) < 10:
        st.warning("Pas assez de données pour la régression multiple (idéalement >= 10).")
    else:
        features = ["ai_knowledge", "internet_access", "training", "trust_ai", "impact_economy"]
        X = df_reg2[features].astype(float)
        y = df_reg2["usage_freq"].astype(float)

        model2 = LinearRegression().fit(X, y)
        y_pred = model2.predict(X)

        r2_2 = r2_score(y, y_pred)
        mse_2 = mean_squared_error(y, y_pred)

        coef_map = {feat: float(c) for feat, c in zip(features, model2.coef_)}
        intercept2 = float(model2.intercept_)

        sorted_abs = sorted(coef_map.items(), key=lambda kv: abs(kv[1]), reverse=True)

        st.write(f"Intercept = **{intercept2:.3f}** | R² = **{r2_2:.3f}** | MSE = **{mse_2:.3f}**")
        fig_bar = px.bar(
            x=[k for k, _ in sorted_abs],
            y=[v for _, v in sorted_abs],
            title="Coefficients (valeur signée) des prédicteurs de la fréquence d’utilisation",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # (3) PCA (réduction des dimensions)
    st.markdown("## Réduction des dimensions : PCA")
    pca_features = [
        "ai_knowledge",
        "internet_access",
        "usage_freq",
        "trust_ai",
        "impact_jobs",
        "impact_education",
        "impact_economy",
        "bias_risk",
        "privacy_risk",
        "misinfo_risk",
        "adoption_intent",
    ]
    df_pca, _, _ = safe_dropna(df, pca_features)

    if len(df_pca) < 8:
        st.warning("Pas assez de données pour PCA.")
    else:
        Xp = df_pca[pca_features].astype(float).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xp)

        pca = PCA(n_components=2, random_state=42)
        comps = pca.fit_transform(Xs)
        explained = pca.explained_variance_ratio_

        st.write(f"Variance expliquée : PC1 = **{explained[0]*100:.1f}%**, PC2 = **{explained[1]*100:.1f}%**")

        df_plot = df_pca.copy()
        df_plot["PC1"] = comps[:, 0]
        df_plot["PC2"] = comps[:, 1]

        fig_pca = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="high_adoption",
            title="PCA (PC1, PC2) — couleur = adoption élevée (high_adoption)",
            labels={"high_adoption": "high_adoption"},
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # (4) Classification supervisée : high_adoption
    st.markdown("## Classification supervisée : prédire high_adoption")
    clf_features = [
        "ai_knowledge",
        "internet_access",
        "training",
        "usage_freq",
        "trust_ai",
        "impact_jobs",
        "impact_education",
        "impact_economy",
        "bias_risk",
        "privacy_risk",
        "misinfo_risk",
    ]
    df_clf, _, _ = safe_dropna(df, clf_features + ["high_adoption"])

    if len(df_clf) < 10:
        st.warning("Pas assez de données pour la classification (ajoute des réponses).")
    else:
        if df_clf["high_adoption"].nunique() < 2:
            st.warning("La classification ne peut pas s’entraîner : une seule classe observée pour high_adoption.")
        else:
            X = df_clf[clf_features].astype(float)
            y = df_clf["high_adoption"].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            models_to_try = []

            logit = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
                ]
            )
            models_to_try.append(("LogisticRegression", logit))

            from sklearn.ensemble import RandomForestClassifier
            rf = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
                ]
            )
            models_to_try.append(("RandomForest", rf))

            best = None
            for name, model in models_to_try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                if best is None or acc > best["accuracy"]:
                    best = {"model_name": name, "accuracy": float(acc), "pred": pred, "y_test": y_test.values}

            st.write(f"Meilleur modèle : **{best['model_name']}**")
            st.write(f"Accuracy : **{best['accuracy']:.2%}**")

            report = classification_report(best["y_test"], best["pred"])
            st.text_area("Classification report", report, height=200)

            cm = confusion_matrix(best["y_test"], best["pred"])
            fig_cm = px.imshow(cm, text_auto=True, title="Matrice de confusion")
            fig_cm.update_xaxes(title="Prédit", tickvals=[0, 1])
            fig_cm.update_yaxes(title="Réel", tickvals=[0, 1])
            st.plotly_chart(fig_cm, use_container_width=True)

    # (5) Classification non-supervisée : KMeans
    st.markdown("## (5) Classification non supervisée : KMeans")
    k_features = [
        "ai_knowledge",
        "internet_access",
        "usage_freq",
        "trust_ai",
        "impact_jobs",
        "impact_education",
        "impact_economy",
        "bias_risk",
        "privacy_risk",
        "misinfo_risk",
        "adoption_intent",
    ]
    df_km, _, _ = safe_dropna(df, k_features)

    if len(df_km) < 10:
        st.warning("Pas assez de données pour KMeans.")
    else:
        Xk = df_km[k_features].astype(float).values
        scaler = StandardScaler()
        Xks = scaler.fit_transform(Xk)

        k = 3
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xks)

        df_km2 = df_km.copy()
        df_km2["cluster"] = labels

        # Visualisation via PCA 2D (juste pour plot)
        pca2 = PCA(n_components=2, random_state=42)
        X2 = pca2.fit_transform(Xks)
        df_km2["PC1"] = X2[:, 0]
        df_km2["PC2"] = X2[:, 1]

        fig_k = px.scatter(
            df_km2,
            x="PC1",
            y="PC2",
            color="cluster",
            title="KMeans (clusters) — visualisation PCA",
            labels={"cluster": "Cluster"},
        )
        st.plotly_chart(fig_k, use_container_width=True)

        st.markdown("### Profils par cluster")
        for c in range(k):
            sub = df_km2[df_km2["cluster"] == c]
            means = sub[k_features].mean(numeric_only=True)

            desc = (
                f"Usage ≈ {means['usage_freq']:.1f}/4 | "
                f"Confiance ≈ {means['trust_ai']:.1f}/5 | "
                f"Risques (biais/priv.) ≈ {means['bias_risk']:.1f}/{means['privacy_risk']:.1f} | "
                f"Adoption intention ≈ {means['adoption_intent']:.1f}/5"
            )
            st.write(f"**Cluster {c}** : {desc}")

# ----------------------------
# STORYTELLING
# ----------------------------
else:
    st.subheader("Storytelling des données (automatique)")

    if df.empty or len(df) < 5:
        st.warning("Pas assez de données. Ajoute des réponses via ‘Collecte’.")
        st.stop()

    # Refaire vite les calculs
    df_reg1, _, _ = safe_dropna(df, ["usage_freq", "ai_knowledge"])
    reg_simple = {"coef_ai_knowledge": None}
    if len(df_reg1) >= 3:
        X = df_reg1[["ai_knowledge"]].values
        y = df_reg1["usage_freq"].values
        model1 = LinearRegression().fit(X, y)
        reg_simple["coef_ai_knowledge"] = float(model1.coef_[0])

    df_reg2, _, _ = safe_dropna(df, ["usage_freq", "ai_knowledge", "internet_access", "training", "trust_ai", "impact_economy"])
    reg_multi = {"top_predictors": []}
    if len(df_reg2) >= 10:
        features = ["ai_knowledge", "internet_access", "training", "trust_ai", "impact_economy"]
        X = df_reg2[features].astype(float)
        y = df_reg2["usage_freq"].astype(float)
        model2 = LinearRegression().fit(X, y)
        coef_map = {feat: float(c) for feat, c in zip(features, model2.coef_)}
        reg_multi["top_predictors"] = sorted(coef_map.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Classification (simple)
    clf_features = [
        "ai_knowledge",
        "internet_access",
        "training",
        "usage_freq",
        "trust_ai",
        "impact_jobs",
        "impact_education",
        "impact_economy",
        "bias_risk",
        "privacy_risk",
        "misinfo_risk",
    ]
    df_clf, _, _ = safe_dropna(df, clf_features + ["high_adoption"])

    clf_results = {"model_name": None, "accuracy": None}
    if len(df_clf) >= 10 and df_clf["high_adoption"].nunique() >= 2:
        X = df_clf[clf_features].astype(float)
        y = df_clf["high_adoption"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
            ]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        clf_results = {"model_name": "LogisticRegression", "accuracy": float(accuracy_score(y_test, pred))}

    # KMeans profils
    k_features = [
        "ai_knowledge",
        "internet_access",
        "usage_freq",
        "trust_ai",
        "impact_jobs",
        "impact_education",
        "impact_economy",
        "bias_risk",
        "privacy_risk",
        "misinfo_risk",
        "adoption_intent",
    ]
    df_km, _, _ = safe_dropna(df, k_features)

    clusters = {"cluster_profiles": {}}
    if len(df_km) >= 10:
        Xk = df_km[k_features].astype(float).values
        scaler = StandardScaler()
        Xks = scaler.fit_transform(Xk)

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(Xks)

        df_km2 = df_km.copy()
        df_km2["cluster"] = labels

        profiles = {}
        for c in range(3):
            sub = df_km2[df_km2["cluster"] == c]
            means = sub[k_features].mean(numeric_only=True)
            desc = (
                f"Usage ≈ {means['usage_freq']:.1f}/4 | "
                f"Confiance ≈ {means['trust_ai']:.1f}/5 | "
                f"Risques (biais/priv.) ≈ {means['bias_risk']:.1f}/{means['privacy_risk']:.1f} | "
                f"Adoption ≈ {means['adoption_intent']:.1f}/5"
            )
            profiles[f"Cluster {c}"] = desc
        clusters["cluster_profiles"] = profiles

    parts = storytelling_parts(df, reg_simple, reg_multi, clf_results, clusters)
    for p in parts:
        st.markdown(p)

st.markdown("---")
st.caption("TP INF232 EC2 — IA Afrique : collecte anonyme + analyse descriptive & ML (régression, PCA, classification, clustering, visualisations).")
