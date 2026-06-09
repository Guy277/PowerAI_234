# 🏗️ Architecture du Projet – PowerAI_234

Voici l'organisation structurelle du projet **Smart Finance Recommender**. Cette architecture a été conçue pour séparer la logique métier, le traitement des données et l'interface utilisateur.

## 📁 Structure des Dossiers

```text
PowerAI_234/
├── 📂 data/                # Persistance et stockage des données
│   ├── clean_data.csv      # Données nettoyées après pipeline Pandas
│   ├── raw_data.csv        # Données brutes (backup/export)
│   ├── users.json          # Base de données principale des clients (Faker)
│   └── users_meta.json     # Métadonnées de versioning des données
│
├── 📂 models/              # Modèles de Machine Learning entraînés (.pkl)
│   ├── XGBoost_pipeline.pkl
│   ├── LGBM_pipeline.pkl
│   └── RandomForest_pipeline.pkl
│
├── 📂 src/                 # Code source de la logique métier
│   └── data_pipeline.py    # Pipeline de traitement et entraînement ML
│
├── 📂 reports/             # Rapports d'analyse et documentation
│   ├── report_ml.md        # Rapport de performance des modèles
│   └── eda_report.html     # Analyse exploratoire des données
│
├── 📄 main.py              # Application principale (Streamlit UI + Logic)
├── 📄 requirements.txt     # Dépendances du projet (Pandas, SciPy, Streamlit...)
├── 📄 .gitignore           # Fichiers exclus du versioning Git
├── 📄 README.md            # Guide général du projet
├── 📄 pitch.md             # Argumentaire pour le jury
└── 📄 architecture.md      # Ce fichier (Vue d'ensemble technique)
```

## ⚙️ Composants Clés

1.  **Interface Utilisateur (main.py)** : Développée avec Streamlit, elle gère la navigation latérale, le formulaire "Nouveau Client" et le dashboard décisionnel.
2.  **Moteur de Recommandation (main.py)** : Basé sur la Programmation Orientée Objet (POO) pour les produits financiers et la Programmation Fonctionnelle (Filter/Map/Reduce) pour le calcul des scores.
3.  **Pipeline de Données (src/data_pipeline.py)** : Utilise Pandas pour le nettoyage (imputation, filtrage d'outliers) et Scikit-Learn pour l'entraînement des modèles prédictifs.
4.  **Analyse Statistique (SciPy)** : Intégrée directement pour valider les hypothèses métier via le test du Chi-Deux ($\chi^2$).
