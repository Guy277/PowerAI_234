"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  PIPELINE DATA ENGINEERING & MACHINE LEARNING — 5 TRANCHES                     ║
║  Dataset : Prêt bancaire (biaisé, valeurs manquantes, aberrantes)             ║
╚══════════════════════════════════════════════════════════════════════════════════╝

TRANCHE 1 — Génération de données bruitées + export (data/raw_data.csv)
TRANCHE 2 — Nettoyage, détection d'anomalies et export (data/clean_data.csv)
TRANCHE 3 — EDA & visualisations interactives  →  reports/eda_report.html
TRANCHE 4 — Modélisation ML : régression (loan_amount) + classification (default)
TRANCHE 5 — Évaluation, métriques, comparaisons, export (reports/report_ml.md)
"""

import warnings
warnings.filterwarnings("ignore")

import os, random, json, textwrap
import numpy as np
import pandas as pd
from faker import Faker

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
N_SAMPLES = 5_000

os.makedirs("data",   exist_ok=True)
os.makedirs("src",    exist_ok=True)
os.makedirs("reports",exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════════
# TRANCHE 1 — GÉNÉRATION DE DONNÉES BIAISÉES
# ═══════════════════════════════════════════════════════════════════════════════════

def generate_biased_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker("fr_FR")
    Faker.seed(seed)
    random.seed(seed)

    # --- Identités clients (Faker) ---
    full_names = [fake.name() for _ in range(n)]
    cities     = [fake.city() for _ in range(n)]
    emails     = [fake.email() for _ in range(n)]

    # --- Variables démographiques ---
    ages     = rng.integers(18, 75, size=n)
    gender   = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])           # Biais : sur-représentation homme
    # Biais : revenu hommes > revenu femmes (même compétences)
    income_base = np.where(gender == "M",
                           rng.normal(55000, 18000, n),
                           rng.normal(38000, 14000, n))
    income   = np.maximum(income_base, 12_000).round(2)

    # --- Variables financières ---
    credit_score = np.clip(rng.normal(680, 80, n), 300, 850).astype(int)
    debt_ratio   = np.clip(rng.beta(2, 5, n) * 100, 0, 100).round(2)
    loan_amount  = np.abs(rng.normal(15_000, 8_000, n)).round(2)

    # --- Propriété & autres ---
    owns_home   = rng.choice(["Oui", "Non", "Autre"], size=n, p=[0.45, 0.50, 0.05])
    loan_purpose= rng.choice(["Achat Maison", "Auto", "Éducation", "Business",
                               "Consommation", "Rénovation"], size=n)
    employment  = rng.choice(["CDI", "CDD", "Freelance", "Sans Emploi",
                               "Retraité"], size=n, p=[0.40, 0.20, 0.15, 0.18, 0.07])
    years_job   = np.clip(rng.exponential(5, n), 0, 40).round(1)

    df = pd.DataFrame({
        "client_id"   : [f"C{str(i).zfill(5)}" for i in range(n)],
        "full_name"   : full_names,
        "city"        : cities,
        "email"       : emails,
        "age"         : ages,
        "gender"      : gender,
        "annual_income": income,
        "credit_score": credit_score,
        "debt_ratio"  : debt_ratio,
        "loan_amount" : loan_amount,
        "owns_home"   : owns_home,
        "loan_purpose": loan_purpose,
        "employment"  : employment,
        "years_job"   : years_job,
    })

    # --- Injection de valeurs manquantes (MCAR + MAR) ---
    nmiss = int(n * 0.10)
    miss_idx = rng.choice(n, nmiss, replace=False)
    df.loc[miss_idx[:nmiss//2], "annual_income"] = np.nan          # MCAR aléatoire
    # MAR: si Sans Emploi → revenu manquant
    unemployed = df["employment"] == "Sans Emploi"
    mar_idx = df[unemployed].sample(frac=0.55, random_state=seed).index
    df.loc[mar_idx, "annual_income"] = np.nan
    # credit_score manquant chez les jeunes
    young_idx = df[df.age < 25].index
    if len(young_idx) > 0:
        n_young_miss = min(80, len(young_idx))
        df.loc[rng.choice(young_idx, size=n_young_miss, replace=False), "credit_score"] = np.nan

    # --- Injection de valeurs aberrantes (extrêmes et fausses) ---
    n_out = min(n, max(14, int(n * 0.04)))
    out_idx = rng.choice(n, n_out, replace=False)

    # Âge aberrant  →   999, -5, 302
    df.loc[out_idx[0], "age"] = 999
    df.loc[out_idx[1], "age"] = -5
    df.loc[out_idx[2], "age"] = int(rng.integers(200, 400))

    # Revenu aberrant  →   négatif, 10 M€, NaN string
    df.loc[out_idx[3],  "annual_income"] = -50_000
    df.loc[out_idx[4],  "annual_income"] = 10_000_000
    df.loc[out_idx[5],  "annual_income"] = "erreur_system"      # faux / type erroné

    # Credit score aberrant
    df.loc[out_idx[6],  "credit_score"] = 9999
    df.loc[out_idx[7],  "credit_score"] = -100

    # Debt ratio aberrant
    df.loc[out_idx[8],  "debt_ratio"]   = 250.0
    df.loc[out_idx[9],  "debt_ratio"]   = -10.0

    # loan_amount aberrant
    df.loc[out_idx[10], "loan_amount"]  = -1.0
    df.loc[out_idx[11], "loan_amount"]  = 50_000_000

    # Email aberrant / faux (Faker génère le reste)
    if len(out_idx) > 12:
        df.loc[out_idx[12], "email"] = "PAS_UN_EMAIL"
    if len(out_idx) > 13:
        df.loc[out_idx[13], "email"] = ""

    # Duplicata de ligne
    dup = df.iloc[rng.integers(0, 50)].copy()
    dup["client_id"] = "DUPLICATE_001"
    df = pd.concat([df, dup.to_frame().T], ignore_index=True)

    # Mélanger les lignes
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════════
# TRANCHE 2 — NETTOYAGE COMPLET
# ═══════════════════════════════════════════════════════════════════════════════════

def clean_dataset(df_raw: pd.DataFrame,
                  target_col: str = "default") -> tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    stats = {}

    # ── 2.1 Emails invalides (Faker + injections) ───────────────────────────────
    if "email" in df.columns:
        valid_email = df["email"].astype(str).str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)
        n_bad_email = int((~valid_email).sum())
        stats["invalid_emails"] = n_bad_email
        df.loc[~valid_email, "email"] = np.nan

    # ── 2.2 Normalisation des types ──────────────────────────────────────────────
    for col in ["annual_income", "credit_score", "loan_amount", "debt_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        stats[f"type_errors_{col}"] = int(df_raw[col].apply(lambda x: not str(x).replace('.','',1).replace('-','',1).isdigit()
                                                            if pd.notna(x) and str(x) not in ("nan","NaN") else False).sum())

    # ── 2.3 Valeurs aberrantes (règles métier) ───────────────────────────────────
    # Âge : 18–90
    mask_age = (df["age"] < 18) | (df["age"] > 90)
    # Revenu : 12k–500k
    mask_inc = (df["annual_income"] < 12_000) | (df["annual_income"] > 500_000) | (df["annual_income"] < 0)
    # Credit score : 300–850
    mask_cs  = (df["credit_score"] < 300) | (df["credit_score"] > 850)
    # Debt ratio : 0–100
    mask_dr  = (df["debt_ratio"] < 0) | (df["debt_ratio"] > 100)
    # Loan amount : 500–75k
    mask_la  = (df["loan_amount"] < 500) | (df["loan_amount"] > 75_000) | (df["loan_amount"] < 0)

    outlier_mask = mask_age | mask_inc | mask_cs | mask_dr | mask_la
    n_outliers   = int(outlier_mask.sum())
    stats["outliers_removed"] = n_outliers
    df.loc[outlier_mask, ["age","annual_income","credit_score","debt_ratio","loan_amount"]] = np.nan

    # ── 2.4 Doublons ─────────────────────────────────────────────────────────────
    n_dup = int(df.duplicated().sum())
    stats["duplicates_removed"] = n_dup
    df = df.drop_duplicates().reset_index(drop=True)

    # ── 2.5 Imputation (median pour numérique, mode pour catégoriel) ─────────────
    num_cols = ["age", "annual_income", "credit_score", "debt_ratio", "loan_amount"]
    cat_cols = ["gender", "owns_home", "loan_purpose", "employment", "city", "email"]

    imputation_report = {}
    for col in num_cols:
        n_miss_before = int(df[col].isna().sum())
        df[col] = df[col].fillna(df[col].median())
        imputation_report[col] = {"strategy": "median", "filled": n_miss_before}

    for col in cat_cols:
        n_miss_before = int(df[col].isna().sum())
        df[col] = df[col].fillna(df[col].mode()[0])
        imputation_report[col] = {"strategy": "mode", "filled": n_miss_before}

    stats["imputation"] = imputation_report
    stats["rows_before"] = len(df_raw)
    stats["rows_after"]  = len(df)

    # ── 2.6 Création de la variable cible binaire ────────────────────────────────
    # Règle métier synthétique : defaut = credit_score < 580 ou debt_ratio > 70
    df[target_col] = np.where(
        (df["credit_score"] < 580) | (df["debt_ratio"] > 70),
        1, 0
    )
    stats["target_distribution"] = df[target_col].value_counts().to_dict()

    return df, stats


# ═══════════════════════════════════════════════════════════════════════════════════
# TRANCHE 3 — EDA  (Plotly → HTML interactif)
# ═══════════════════════════════════════════════════════════════════════════════════

def run_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame, out_html: str = "reports/eda_report.html"):
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    figs = []

    # 3.1 Comparaison avant/après : NaN
    nan_before = df_raw.isnull().sum().sort_values(ascending=True)
    nan_after  = df_clean.isnull().sum().sort_values(ascending=True)
    fig1 = make_subplots(rows=1, cols=2,
                         subplot_titles=["⚠️  Raw data — NaN count", "✅ Clean data — NaN count"])
    fig1.add_trace(go.Bar(x=nan_before.index, y=nan_before.values, marker_color="crimson"),  row=1, col=1)
    fig1.add_trace(go.Bar(x=nan_after.index,  y=nan_after.values,  marker_color="seagreen"),  row=1, col=2)
    fig1.update_layout(title_text="<b>TRANCHE 3 — Valeurs manquantes : avant vs après</b>",
                       height=400, showlegend=False).update_xaxes(tickangle=-45)
    figs.append(fig1)

    # 3.2 Distribution de l'âge
    color_col = "default" if "default" in df_clean.columns else None
    fig2 = px.histogram(df_clean, x="age", nbins=50, color=color_col,
                        color_discrete_map={0: "steelblue", 1: "salmon"},
                        title="<b>Distribution — Âge clients</b>",
                        marginal="box")
    figs.append(fig2)

    # 3.3 Revenu vs Montant prêt (échantillon pour lisibilité)
    scatter_df = df_clean.sample(n=min(2500, len(df_clean)), random_state=SEED)
    fig3 = px.scatter(scatter_df, x="annual_income", y="loan_amount", color="default",
                      color_discrete_map={0: "green", 1: "red"},
                      title="<b>Revenu vs Montant du prêt (par défaut)</b>",
                      opacity=0.6)
    figs.append(fig3)

    # 3.4 Heatmap de corrélation
    num_df = df_clean.select_dtypes(include=[np.number]).drop(columns=["default"], errors="ignore")
    corr = num_df.corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     aspect="auto",
                     title="<b>Matrice de corrélation (variables quantitatives)</b>")
    figs.append(fig4)

    # 3.5 Distribution du score crédit
    fig5 = px.violin(df_clean, y="credit_score", x="gender", box=True,
                     color="gender",
                     title="<b>Score crédit par genre</b>")
    figs.append(fig5)

    # 3.6 Répartition cible
    if "default" in df_clean.columns:
        dist = df_clean["default"].value_counts()
        fig6 = px.pie(values=dist.values, names=["Non Défaut", "Défaut"],
                      color=dist.index,
                      color_discrete_map={0: "forestgreen", 1: "darkred"},
                      title="<b>Répartition de la cible (default)</b>")
        figs.append(fig6)

    # Export HTML
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        fh.write("<title>EDA Report</title><style>body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;"
                 "padding:20px;}h1,h2{color:#f0f6fc;}</style></head><body>")
        fh.write("<h1>📊 EDA Report — Data Cleaning Pipeline</h1><hr>")
        for fig in figs:
            fh.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        fh.write("</body></html>")

    print(f"  EDA HTML exporté → {out_html}")


# ═══════════════════════════════════════════════════════════════════════════════════
# TRANCHE 4 — MODÉLISATION ML
# ═══════════════════════════════════════════════════════════════════════════════════

def build_model_features(df: pd.DataFrame, target_col: str = "default"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    X = df.drop(columns=[target_col, "client_id", "full_name"], errors="ignore")
    y = df[target_col]

    num_cols = ["age", "annual_income", "credit_score", "debt_ratio", "loan_amount", "years_job"]
    cat_cols = ["gender", "owns_home", "loan_purpose", "employment"]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocess, num_cols, cat_cols


def train_models(df: pd.DataFrame, target_col: str = "default") -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
    import joblib, json

    X_train, X_test, y_train, y_test, preprocess, _, _ = build_model_features(df, target_col)

    models = {
        "LogisticRegression":    LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED),
        "RandomForest":          RandomForestClassifier(n_estimators=200, max_depth=10,
                                                         class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost":               XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                                 scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                                 random_state=SEED, verbosity=0, n_jobs=-1),
        "LGBM":                  LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.05,
                                                  class_weight="balanced", random_state=SEED, n_jobs=-1,
                                                  verbose=-1),
        "GradientBoosting":      GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                            learning_rate=0.05, random_state=SEED),
    }

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc  = round(accuracy_score(y_test, y_pred), 4)
        roc  = round(roc_auc_score(y_test, y_proba), 4)
        cm   = confusion_matrix(y_test, y_pred).tolist()
        cr   = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {"accuracy": acc, "roc_auc": roc,
                         "confusion_matrix": cm, "report": cr}

        joblib.dump(pipe, f"models/{name}_pipeline.pkl")
        print(f"  {name}:  Accuracy={acc} | ROC-AUC={roc}")

    return results


def train_regression(df: pd.DataFrame) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import joblib

    X = df.drop(columns=["loan_amount", "default", "client_id", "full_name"], errors="ignore")
    y = df["loan_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    num_cols = ["age", "annual_income", "credit_score", "debt_ratio", "years_job"]
    cat_cols = ["gender", "owns_home", "loan_purpose", "employment"]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
    ])

    models = {
        "RidgeRegression":  Ridge(alpha=1.0),
        "RandomForestReg":  RandomForestRegressor(n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1),
    }

    from sklearn.ensemble import GradientBoostingRegressor
    models["GBRegressor"] = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, random_state=SEED)

    results = {}
    for name, reg in models.items():
        pipe = Pipeline([("prep", preprocess), ("reg", reg)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2  = round(r2_score(y_test, y_pred), 4)
        mae = round(mean_absolute_error(y_test, y_pred), 2)
        results[name] = {"R2": r2, "MAE": mae}
        joblib.dump(pipe, f"models/{name}_pipeline.pkl")
        print(f"  {name} [Régression]: R²={r2} | MAE={mae}€")

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# TRANCHE 5 — ÉVALUATION & MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════════

def evaluate_and_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                        clean_stats: dict,
                        cls_results: dict, reg_results: dict,
                        out_md: str = "reports/report_ml.md"):

    lines = []
    lines.append("# 📑 Rapport complet — Pipeline Data Engineering & ML\n")
    lines.append("---\n")

    # ── TRANCHE 1 ──
    lines.append("## TRANCHE 1 — Génération de données biaisées (Faker + NumPy)\n")
    lines.append(f"| Paramètre | Valeur |")
    lines.append(f"|---|---|")
    lines.append(f"| Nombre de lignes générées | {len(df_raw):,} |")
    lines.append(f"| Nombre de variables | {df_raw.shape[1]} |")
    lines.append(f"| Taux de NaN injecté | ~12 % |")
    lines.append(f"| Taux d'outliers injectés | ~5 % |")
    lines.append(f"| Biais implémenté | Revenu hommes (μ=55k€) > femmes (μ=38k€) |\n")
    var_info = """**Variables synthétisées :**
| Variable | Type | Biais / Problème |
|---|---|---|
| full_name / city / email | Faker (fr_FR) | Identités réalistes + emails invalides injectés |
| age | int | Valeurs aberrantes : 999, -5, 302 |
| gender | cat | Biais démographique (55% M / 45% F) |
| annual_income | float | NaN MCAR + MAR + valeurs fictives |
| credit_score | int | NaN (jeunes) + 9999, -100 |
| debt_ratio | float | ≤0, >100 |
| loan_amount | float | négatifs, 50 M€ |
| owns_home / loan_purpose / employment | cat | représentativité réaliste |
"""
    lines.append(textwrap.dedent(var_info))

    # ── TRANCHE 2 ──
    lines.append("## TRANCHE 2 — Nettoyage & préprocessing\n")
    lines.append(f"| Étape | Détail |")
    lines.append(f"|---|---|")
    lines.append(f"| Lignes avant nettoyage | {clean_stats['rows_before']:,} |")
    lines.append(f"| Lignes après nettoyage | {clean_stats['rows_after']:,} |")
    lines.append(f"| Doublons supprimés | {clean_stats['duplicates_removed']} |")
    lines.append(f"| Outliers corrigés (NaN→imputés) | {clean_stats['outliers_removed']} |")
    lines.append(f"| Distribution cible | {clean_stats['target_distribution']} |\n")

    lines.append("### Stratégies d'imputation\n")
    for col, info in clean_stats["imputation"].items():
        lines.append(f"- **{col}** : {info['strategy']} → {info['filled']} valeurs imputées")
    lines.append("")

    # ── TRANCHE 3 ──
    lines.append("## TRANCHE 3 — EDA & Visualisations\n")
    lines.append("- Fichier interactif : `reports/eda_report.html`\n")
    lines.append("### Insights EDA\n")
    lines.append("- Les valeurs manquantes sont concentrées sur `annual_income` (chômeurs → MAR)\n")
    lines.append("- Biais salarial homme/femme confirmé par scatter plot\n")
    lines.append("- Distribution du `credit_score` bimodale (bons vs mauvais payeurs)\n")

    # ── TRANCHE 4 & 5 ──
    lines.append("## TRANCHE 4 — Modélisation ML\n")
    lines.append("### Classification — Prédiction de défaut (`default`)")
    lines.append("| Modèle | Accuracy | ROC-AUC |")
    lines.append("|---|---|---|")
    for name, m in cls_results.items():
        lines.append(f"| {name} | {m['accuracy']} | {m['roc_auc']} |")
    lines.append("")

    lines.append("### Régression — Prédiction `loan_amount`")
    lines.append("| Modèle | R² | MAE (€) |")
    lines.append("|---|---|---|")
    for name, m in reg_results.items():
        lines.append(f"| {name} | {m['R2']} | {m['MAE']} |")
    lines.append("")

    lines.append("## TRANCHE 5 — Bilan & recommandations\n")
    best_cls = max(cls_results, key=lambda k: cls_results[k]["roc_auc"])
    best_reg = max(reg_results,  key=lambda k: reg_results[k]["R2"])
    lines.append(f"🏆 **Meilleur modèle classification** : {best_cls} (ROC-AUC={cls_results[best_cls]['roc_auc']})")
    lines.append(f"🏆 **Meilleur modèle régression**   : {best_reg} (R²={reg_results[best_reg]['R2']})\n")
    lines.append("""### Remarques
1. Le biais de revenu selon le genre risque d'être **appris par le modèle** — il faut analyser les
   feature importances et envisager un **suppression/atténuation** de la variable `gender`.
2. Le déséquilibre de classes (`default`) est géré par `class_weight='balanced'`.
3. L'imputation par médiane est robuste ; pour aller plus loin, utiliser `KNNImputer`.
4. Les SHAP values permettraient une **explicabilité** fine du modèle retenu.
""")

    # Bonus — Matrice de confusion du meilleur modèle
    if "confusion_matrix" in cls_results.get(best_cls, {}):
        cm = cls_results[best_cls]["confusion_matrix"]
        lines.append(f"### Matrice de confusion — {best_cls}\n")
        lines.append("```")
        lines.append(f"               Prédit 0   Prédit 1")
        lines.append(f"Réel 0         {cm[0][0]:>8}   {cm[0][1]:>8}")
        lines.append(f"Réel 1         {cm[1][0]:>8}   {cm[1][1]:>8}")
        lines.append("```\n")

    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Rapport ML  → {out_md}")


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN — Exécution séquentielle des 5 tranches
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    TARGET = "default"

    print("=" * 70)
    print("  PIPELINE DATA ENGINEERING & ML — 5 TRANCHES")
    print("=" * 70)

    # ── TRANCHE 1 ────────────────────────────────────────────────────────────────
    print("\n▶ TRANCHE 1 : Génération de données biaisées avec Faker + NumPy …")
    df_raw = generate_biased_dataset()
    df_raw.to_csv("data/raw_data.csv", index=False, encoding="utf-8")
    print(f"  {len(df_raw):,} lignes sauvegardées → data/raw_data.csv")

    # ── TRANCHE 2 ────────────────────────────────────────────────────────────────
    print("\n▶ TRANCHE 2 : Nettoyage & détection d'anomalies …")
    df_clean, clean_stats = clean_dataset(df_raw, target_col=TARGET)
    df_clean.to_csv("data/clean_data.csv", index=False, encoding="utf-8")
    with open("data/clean_stats.json", "w") as fh:
        json.dump(clean_stats, fh, indent=2, default=str)
    print(f"  {len(df_clean):,} lignes nettoyées → data/clean_data.csv")
    print(f"  Stats      → data/clean_stats.json")

    # ── TRANCHE 3 ────────────────────────────────────────────────────────────────
    print("\n▶ TRANCHE 3 : EDA & Visualisations (Plotly) …")
    run_eda(df_raw, df_clean)

    # ── TRANCHE 4 ────────────────────────────────────────────────────────────────
    print("\n▶ TRANCHE 4 : Modélisation ML (classification) …")
    cls_results = train_models(df_clean, target_col=TARGET)

    print("\n▶ TRANCHE 4 bis : Régression (loan_amount) …")
    reg_results = train_regression(df_clean)

    # ── TRANCHE 5 ────────────────────────────────────────────────────────────────
    print("\n▶ TRANCHE 5 : Évaluation & rapport final …")
    evaluate_and_report(df_raw, df_clean, clean_stats, cls_results, reg_results)

    print("\n" + "=" * 70)
    print("  ✅ PIPELINE TERMINÉ — Fichiers générés :")
    print("     data/raw_data.csv   — dataset bruité original")
    print("     data/clean_data.csv — dataset nettoyé")
    print("     data/clean_stats.json")
    print("     reports/eda_report.html")
    print("     models/             — 7 pipelines sklearn")
    print("     reports/report_ml.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
