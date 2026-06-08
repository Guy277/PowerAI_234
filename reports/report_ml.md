# 📑 Rapport complet — Pipeline Data Engineering & ML

---

## TRANCHE 1 — Génération de données biaisées (Faker + NumPy)

| Paramètre | Valeur |
|---|---|
| Nombre de lignes générées | 5,001 |
| Nombre de variables | 14 |
| Taux de NaN injecté | ~12 % |
| Taux d'outliers injectés | ~5 % |
| Biais implémenté | Revenu hommes (μ=55k€) > femmes (μ=38k€) |

**Variables synthétisées :**
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

## TRANCHE 2 — Nettoyage & préprocessing

| Étape | Détail |
|---|---|
| Lignes avant nettoyage | 5,001 |
| Lignes après nettoyage | 5,001 |
| Doublons supprimés | 0 |
| Outliers corrigés (NaN→imputés) | 42 |
| Distribution cible | {0: 4455, 1: 546} |

### Stratégies d'imputation

- **age** : median → 42 valeurs imputées
- **annual_income** : median → 774 valeurs imputées
- **credit_score** : median → 122 valeurs imputées
- **debt_ratio** : median → 42 valeurs imputées
- **loan_amount** : median → 42 valeurs imputées
- **gender** : mode → 0 valeurs imputées
- **owns_home** : mode → 0 valeurs imputées
- **loan_purpose** : mode → 0 valeurs imputées
- **employment** : mode → 0 valeurs imputées
- **city** : mode → 0 valeurs imputées
- **email** : mode → 2 valeurs imputées

## TRANCHE 3 — EDA & Visualisations

- Fichier interactif : `reports/eda_report.html`

### Insights EDA

- Les valeurs manquantes sont concentrées sur `annual_income` (chômeurs → MAR)

- Biais salarial homme/femme confirmé par scatter plot

- Distribution du `credit_score` bimodale (bons vs mauvais payeurs)

## TRANCHE 4 — Modélisation ML

### Classification — Prédiction de défaut (`default`)
| Modèle | Accuracy | ROC-AUC |
|---|---|---|
| LogisticRegression | 0.9001 | 0.9551 |
| RandomForest | 1.0 | 1.0 |
| XGBoost | 0.994 | 0.9997 |
| LGBM | 0.998 | 0.9999 |
| GradientBoosting | 1.0 | 1.0 |

### Régression — Prédiction `loan_amount`
| Modèle | R² | MAE (€) |
|---|---|---|
| RidgeRegression | -0.0093 | 6121.36 |
| RandomForestReg | -0.0162 | 6146.28 |
| GBRegressor | -0.0263 | 6182.71 |

## TRANCHE 5 — Bilan & recommandations

🏆 **Meilleur modèle classification** : RandomForest (ROC-AUC=1.0)
🏆 **Meilleur modèle régression**   : RidgeRegression (R²=-0.0093)

### Remarques
1. Le biais de revenu selon le genre risque d'être **appris par le modèle** — il faut analyser les
   feature importances et envisager un **suppression/atténuation** de la variable `gender`.
2. Le déséquilibre de classes (`default`) est géré par `class_weight='balanced'`.
3. L'imputation par médiane est robuste ; pour aller plus loin, utiliser `KNNImputer`.
4. Les SHAP values permettraient une **explicabilité** fine du modèle retenu.

### Matrice de confusion — RandomForest

```
               Prédit 0   Prédit 1
Réel 0              892          0
Réel 1                0        109
```
