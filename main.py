# ==============================================================================
# ============================================================
# FICHIER PRINCIPAL : main.py
# PROJET             : PowerAI_234 – Smart Finance Recommender
# BOOTCAMP           : COT_GenAI & Machine Learning 2026
# DESCRIPTION        : Application Streamlit simulant un conseiller bancaire
#                      intelligent basé sur la POO, la programmation fonctionnelle
#                      et la data science (NumPy, Pandas, SciPy, Matplotlib, Seaborn).
# ============================================================
# ==============================================================================


# ==============================================================================
# ÉTAPE 0 : IMPORTATION DES BIBLIOTHÈQUES / LIBRAIRIES
# ==============================================================================

# --- streamlit : Framework web interactif pour Python ---
# Permet de construire des interfaces graphiques (UI) directement en Python,
# sans avoir besoin d'HTML/JavaScript séparés. Chaque widget (bouton, slider,
# tableau…) se met à jour en temps réel dès que l'utilisateur interagit.
import streamlit as st

# --- pandas : Manipulation de données tabulaires ---
# Fournit la structure "DataFrame" (tableau 2D avec colonnes nommées),
# similaire à une feuille Excel mais manipulable par du code Python.
# Utilisé ici pour charger, nettoyer et filtrer les profils clients.
import pandas as pd

# --- numpy : Calcul numérique et tableaux multidimensionnels ---
# Bibliothèque fondamentale de la data science Python. Offre des tableaux
# (arrays) très rapides, des fonctions mathématiques (distributions
# statistiques, clip, seed…) et des opérations vectorisées sur des données.
import numpy as np

# --- scipy.stats : Statistiques avancées ---
# Module de SciPy dédié aux tests statistiques (Chi-deux, loi normale…).
# Utilisé pour valider scientifiquement les distributions de la population
# et calculer les centiles de salaire (CDF de la loi normale).
import scipy.stats as stats

# --- scipy.spatial.distance : Calcul de distances entre vecteurs ---
# Sous-module de SciPy permettant de calculer des distances (euclidienne,
# cosinus, Manhattan…) entre des points dans un espace multidimensionnel.
# Utilisé pour mesurer la similarité entre les profils financiers des clients.
from scipy.spatial import distance

# --- matplotlib.pyplot : Tracé de graphiques scientifiques ---
# Bibliothèque de visualisation de référence en Python, inspirée de MATLAB.
# Fournit les fonctions de base pour créer des figures, des axes, des courbes
# et des images. Utilisée ici pour composer le dashboard 2x2 enregistré en PNG.
import matplotlib.pyplot as plt

# --- seaborn : Visualisation statistique haut niveau ---
# Bibliothèque de visualisation construite par-dessus Matplotlib. Offre des
# graphiques statistiques clé-en-main (histogramme + KDE, heatmap, barplot…)
# avec des palettes de couleurs soignées et un code plus concis que Matplotlib pur.
import seaborn as sns

# --- json : Sérialisation et désérialisation JSON ---
# Module standard Python pour lire et écrire des fichiers au format JSON
# (JavaScript Object Notation). Utilisé pour stocker et charger les profils
# clients dans le fichier `data/users.json`.
import json
import io

# --- os : Interaction avec le système de fichiers ---
# Module standard Python donnant accès aux fonctionnalités du système
# d'exploitation : lecture de chemins de fichiers, création de dossiers,
# vérification d'existence de fichiers, etc.
import os

# --- random : Génération de nombres aléatoires (Python standard) ---
# Module standard Python pour le tirage aléatoire "classique" (choix dans une
# liste, nombre flottant uniforme…). Utilisé conjointement avec NumPy pour
# mélanger les noms africains et français dans la génération synthétique.
import random

# --- functools.reduce : Réduction de liste par accumulation ---
# Fonction importée du module `functools` qui applique une fonction binaire
# de manière cumulative sur les éléments d'une liste pour la "réduire" à
# une seule valeur (ou une seule structure, ici un dictionnaire groupé).
from functools import reduce

# --- faker : Génération de données fictives réalistes ---
# Bibliothèque tierce qui produit des noms, adresses, e-mails, dates…
# aléatoires mais plausibles. La locale 'fr_FR' génère des données en
# français. Utilisée pour créer la base de clients fictifs de PowerBank.
from faker import Faker


# ==============================================================================
# ÉTAPE 1 : CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================
# Configure les métadonnées globales de l'application web :

st.set_page_config(
    page_title="PowerBank - Smart Finance Recommender",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# ÉTAPE 2 : DÉFINITION DES CHEMINS DE FICHIERS
# ==============================================================================
# On construit tous les chemins de manière dynamique et portable grâce à os.path.
# Ainsi, le projet fonctionne quel que soit le système d'exploitation ou le
# répertoire d'installation.

# Dossier racine du projet (là où se trouve ce fichier main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sous-dossier "data/" qui contiendra les fichiers générés
DATA_DIR = os.path.join(BASE_DIR, "data")

# Chemin complet vers le fichier JSON des profils clients
USERS_JSON_PATH = os.path.join(DATA_DIR, "users.json")
USERS_META_PATH = os.path.join(DATA_DIR, "users_meta.json")
USERS_DATA_VERSION = 2  # Incrémenter pour forcer la régénération avec plus d'anomalies

# Chemin complet vers l'image du dashboard généré par Matplotlib
DASHBOARD_PNG_PATH = os.path.join(DATA_DIR, "dashboard.png")

# Crée le dossier "data/" s'il n'existe pas encore (évite une erreur FileNotFound)
os.makedirs(DATA_DIR, exist_ok=True)

# Palettes thème sombre / clair (lisibilité texte + graphiques)
APP_THEMES = {
    "dark": {
        "bg_main": "#0E1117",
        "bg_card": "#161B22",
        "bg_code": "#0D1117",
        "text": "#FFFFFF",
        "text_muted": "#C9D1D9",
        "text_subtle": "#8B949E",
        "border": "#21262D",
        "border_strong": "#30363D",
        "accent": "#58A6FF",
        "plotly_template": "plotly_dark",
        "plot_paper": "#1C2128",
        "plot_bg": "#22272E",
        "plot_font": "#E6EDF3",
        "plot_title": "#FFFFFF",
        "plot_grid": "#30363D",
        "anomaly_null": ("#5c3535", "#ffecec"),
        "anomaly_neg": ("#5c4a1f", "#fff3c4"),
        "anomaly_goal": ("#3d2b5c", "#e8d4ff"),
        "anomaly_risk": ("#2b4a3d", "#c8ffe0"),
        "anomaly_type": ("#1f3d5c", "#c8e4ff"),
    },
    "light": {
        "bg_main": "#FFFFFF",
        "bg_card": "#F6F8FA",
        "bg_code": "#F6F8FA",
        "text": "#1F2328",
        "text_muted": "#424A53",
        "text_subtle": "#656D76",
        "border": "#D0D7DE",
        "border_strong": "#C9D1D9",
        "accent": "#0969DA",
        "plotly_template": "plotly_white",
        "plot_paper": "#FFFFFF",
        "plot_bg": "#F6F8FA",
        "plot_font": "#1F2328",
        "plot_title": "#1F2328",
        "plot_grid": "#D0D7DE",
        "anomaly_null": ("#ffe0e0", "#8b0000"),
        "anomaly_neg": ("#fff3cd", "#7a5c00"),
        "anomaly_goal": ("#ede7f6", "#4a148c"),
        "anomaly_risk": ("#e8f5e9", "#1b5e20"),
        "anomaly_type": ("#e3f2fd", "#0d47a1"),
    },
}


def get_app_theme():
    return st.session_state.get("app_theme", "dark")


def theme_palette(theme=None):
    return APP_THEMES[theme or get_app_theme()]


def inject_theme_css(theme=None):
    """Injecte le CSS global selon le thème actif."""
    t = theme_palette(theme)
    st.markdown(f"""
    <style>
        .main {{
            background-color: {t['bg_main']};
            color: {t['text']};
        }}
        .stMetric {{
            background-color: {t['bg_card']};
            padding: 15px;
            border-radius: 10px;
            border: 1px solid {t['border']};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        }}
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] p {{
            color: {t['text_muted']} !important;
            font-size: 14px;
            font-weight: 500;
        }}
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] > div {{
            color: {t['accent']} !important;
            font-size: 24px;
            font-weight: bold;
        }}
        div[data-testid="stMetricDelta"] {{
            color: {t['text_subtle']} !important;
        }}
        .recom-card {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border_strong']};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
        }}
        .recom-card:hover {{
            border-color: {t['accent']};
        }}
        .code-box {{
            background-color: {t['bg_code']};
            border: 1px solid {t['border']};
            border-radius: 8px;
            padding: 15px;
        }}
        .badge-premium {{
            background-color: #f39c12; color: #000000;
            padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold;
        }}
        .badge-standard {{
            background-color: {t['accent']}; color: #ffffff;
            padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold;
        }}
        .badge-basic {{
            background-color: {t['text_subtle']}; color: #ffffff;
            padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold;
        }}
        [data-testid="stSidebar"] {{
            background-color: {t['bg_card']};
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {t['text']};
        }}
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# ÉTAPE 3 : DÉFINITION DES CLASSES DU MODÈLE (POO)
#           Concepts démontrés : Encapsulation, Héritage, Polymorphisme
# ==============================================================================

class User:
    """
    Classe représentant un client de PowerBank.
    Démontre l'encapsulation via des attributs privés et des getters.

    L'encapsulation consiste à masquer les données internes d'un objet
    (attributs privés préfixés par __) et à n'y donner accès qu'au travers
    de méthodes contrôlées (les @property / getters), garantissant ainsi
    l'intégrité des données.
    """
    def __init__(self, name, age, salary, savings, financial_goal, risk_level, bank_status):
        """
        Constructeur : initialise tous les attributs du client.
        Le double underscore (__) rend chaque attribut PRIVÉ :
        il n'est pas accessible directement depuis l'extérieur de la classe.
        """
        self.__name = name                   # Prénom + Nom du client
        self.__age = int(age)                # Âge (converti en entier)
        self.__salary = float(salary)        # Salaire mensuel en FCFA
        self.__savings = float(savings)      # Épargne totale en FCFA
        self.__financial_goal = financial_goal  # Objectif : retirement / housing / emergency / investment
        self.__risk_level = risk_level          # Tolérance au risque : low / medium / high
        self.__bank_status = bank_status        # Statut client : basic / standard / premium

    # --- Getters : propriétés publiques en lecture seule ---
    # Le décorateur @property transforme une méthode en attribut "calculé",
    # rendant l'accès en lecture propre (user.name au lieu de user.get_name()).

    @property
    def name(self):
        """Retourne le nom complet du client (lecture seule)."""
        return self.__name

    @property
    def age(self):
        """Retourne l'âge du client (lecture seule)."""
        return self.__age

    @property
    def salary(self):
        """Retourne le salaire mensuel du client (lecture seule)."""
        return self.__salary

    @property
    def savings(self):
        """Retourne l'épargne totale du client (lecture seule)."""
        return self.__savings

    @property
    def financial_goal(self):
        """Retourne l'objectif financier principal du client (lecture seule)."""
        return self.__financial_goal

    @property
    def risk_level(self):
        """Retourne le niveau de tolérance au risque du client (lecture seule)."""
        return self.__risk_level

    @property
    def bank_status(self):
        """Retourne le statut bancaire du client (lecture seule)."""
        return self.__bank_status


class Recommender(User):
    """
    Classe héritant de User et y ajoutant la logique de recommandation.

    L'HÉRITAGE permet à Recommender de réutiliser tous les attributs et
    méthodes de User sans les réécrire (via super().__init__(...)).

    Le POLYMORPHISME est illustré par le fait que la méthode recommend()
    adapte son comportement dynamiquement selon le profil du client,
    sans que l'appelant ne sache quel type de produit est traité.
    """
    def __init__(self, name, age, salary, savings, financial_goal, risk_level, bank_status):
        """
        Constructeur : appelle le constructeur de la classe parente (User)
        via super() pour initialiser les attributs hérités.
        """
        super().__init__(name, age, salary, savings, financial_goal, risk_level, bank_status)

    def recommend(self, catalog):
        """
        Génère des recommandations personnalisées à partir du catalogue de produits financiers
        en utilisant les fonctions d'ordre supérieur : filter(), map(), reduce() et lambda.

        Paramètre :
            catalog (list) : Liste d'objets FinancialProduct à évaluer.

        Retour :
            dict : Dictionnaire groupé par catégorie, chaque valeur étant
                   une liste triée de recommandations (produit, score, raisons).
        """
        # ---- ÉTAPE A : FILTER – Sélectionner les produits éligibles ----
        # filter() parcourt chaque produit du catalogue et ne conserve que
        # ceux pour lesquels la méthode is_eligible(self) retourne True.
        # La lambda remplace une fonction nommée : lambda p = une fonction anonyme
        # qui prend un produit "p" et appelle p.is_eligible(self).
        eligible_products = list(filter(lambda p: p.is_eligible(self), catalog))

        # ---- ÉTAPE B : MAP – Enrichir chaque produit éligible ----
        # map() applique une transformation à chaque élément de la liste.
        # Ici, chaque produit est converti en dictionnaire contenant :
        #   - "product" : l'objet produit lui-même
        #   - "score"   : l'indice de correspondance calculé (0 à 100)
        #   - "reasons" : la liste des justifications textuelles
        recommendations = list(map(lambda p: {
            "product": p,
            "score": p.recommend_score(self),
            "reasons": p.get_reason(self)
        }, eligible_products))

        # Tri des recommandations par score décroissant (meilleur en premier)
        recommendations.sort(key=lambda r: r["score"], reverse=True)

        # ---- ÉTAPE C : REDUCE – Regrouper par catégorie de produit ----
        # reduce() parcourt la liste et accumule les résultats dans un seul
        # dictionnaire. Ici, "accumulator" est ce dictionnaire et
        # "recommendation_item" est l'élément courant de la liste.
        def group_by_category(accumulator, recommendation_item):
            """
            Fonction d'accumulation pour reduce() :
            regroupe les recommandations par clé de catégorie (banking,
            insurance, investment, loans).
            """
            category = recommendation_item["product"].category
            # Créer la liste pour cette catégorie si elle n'existe pas encore
            if category not in accumulator:
                accumulator[category] = []
            accumulator[category].append(recommendation_item)
            return accumulator

        # reduce(fonction, itérable, valeur_initiale)
        # Commence avec un dict vide {} et le remplit catégorie par catégorie.
        grouped_recommendations = reduce(group_by_category, recommendations, {})
        return grouped_recommendations


# ==============================================================================
# ÉTAPE 4 : HIÉRARCHIE DES PRODUITS FINANCIERS (POLYMORPHISME & HÉRITAGE)
# ==============================================================================
# Chaque produit hérite de FinancialProduct et surcharge (override) les
# méthodes is_eligible(), recommend_score() et get_reason() selon ses règles
# métier spécifiques. C'est le principe du Polymorphisme.

class FinancialProduct:
    """
    Classe de base (classe mère) pour tous les produits financiers de PowerBank.
    Définit l'interface commune (contrat) que toutes les sous-classes doivent respecter.
    Les comportements par défaut (éligibilité universelle, score = 50) sont surchargés
    dans chaque sous-classe pour refléter les règles réelles du produit.
    """
    def __init__(self, name, category, description):
        """
        Paramètres :
            name        (str) : Nom commercial du produit
            category    (str) : Catégorie : banking / insurance / investment / loans
            description (str) : Description courte affichée à l'utilisateur
        """
        self.name = name
        self.category = category  # Catégorie : banking, insurance, investment, loans
        self.description = description

    def is_eligible(self, user: User) -> bool:
        """
        Méthode polymorphe : vérifie si le client est éligible à ce produit.
        Par défaut : tous les clients sont éligibles (True).
        Les sous-classes surchargent cette méthode pour ajouter des conditions.
        """
        return True

    def recommend_score(self, user: User) -> float:
        """
        Méthode polymorphe : calcule l'indice d'adéquation du produit au client (0 à 100).
        Par défaut : score neutre de 50. Les sous-classes ajustent selon le profil.
        """
        return 50.0

    def get_reason(self, user: User) -> list:
        """
        Méthode polymorphe : retourne la liste des justifications textuelles
        expliquant pourquoi ce produit est recommandé au client.
        """
        return ["Produit bancaire standard adapté à votre profil."]


# ==============================================================================
# SOUS-SECTION 4.1 : PRODUITS BANCAIRES (catégorie "banking")
# ==============================================================================

class SavingsAccount(FinancialProduct):
    """Compte d'épargne rémunéré – produit privilégié pour la constitution d'un fonds."""
    def __init__(self):
        # Appel du constructeur parent pour renseigner nom, catégorie et description
        super().__init__(
            "Compte Épargne PowerSave",
            "banking",
            "Compte d'épargne rémunéré avec disponibilité immédiate des fonds."
        )

    def recommend_score(self, user: User) -> float:
        """
        Calcule le score d'adéquation :
        - Score de base : 60
        - +30 si l'objectif est "emergency" (fonds d'urgence)
        - +15 si l'objectif est "retirement" (retraite)
        - +10 si l'épargne actuelle est faible (< 50 000 FCFA)
        """
        score = 60.0
        if user.financial_goal == 'emergency':
            score += 30.0
        elif user.financial_goal == 'retirement':
            score += 15.0
        if user.savings < 50000:
            score += 10.0
        # min() plafonne le score à 100 pour ne pas dépasser le maximum
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        """Retourne les raisons personnalisées de la recommandation."""
        reasons = ["Taux d'intérêt avantageux pour sécuriser votre épargne."]
        if user.financial_goal == 'emergency':
            reasons.append("Recommandé en priorité pour constituer votre fonds d'urgence sécurisé.")
        if user.savings < 50000:
            reasons.append("Idéal pour commencer à bâtir votre épargne de base.")
        return reasons


class CurrentAccount(FinancialProduct):
    """Compte courant pour la gestion quotidienne des finances – produit universel."""
    def __init__(self):
        super().__init__(
            "Compte Courant PowerCash",
            "banking",
            "Compte de dépôt pour gérer vos finances au quotidien avec carte bancaire incluse."
        )

    def recommend_score(self, user: User) -> float:
        """
        Score de base élevé (80) car c'est un produit fondamental.
        Bonus selon le statut : +15 premium, +5 standard.
        """
        score = 80.0
        if user.bank_status == 'premium':
            score += 15.0
        elif user.bank_status == 'standard':
            score += 5.0
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Indispensable pour domicilier vos revenus et gérer vos transactions."]
        if user.bank_status == 'premium':
            reasons.append("Avantages exclusifs PowerGold et frais de transaction offerts à l'international.")
        return reasons


class MicroSaving(FinancialProduct):
    """Option d'épargne automatique par arrondi – cible les jeunes et petits budgets."""
    def __init__(self):
        super().__init__(
            "Option Micro-Épargne PowerFlex",
            "banking",
            "Arrondi automatique de vos achats à l'unité supérieure pour épargner sans effort."
        )

    def is_eligible(self, user: User) -> bool:
        """
        Éligible uniquement si :
        - Statut "basic" ou "standard" (pas premium, déjà mieux doté), OU
        - Client âgé de moins de 30 ans (jeune actif / étudiant)
        """
        return user.bank_status in ['basic', 'standard'] or user.age < 30

    def recommend_score(self, user: User) -> float:
        """Score de base 65 ; bonus fort pour les moins de 25 ans et statut basic."""
        score = 65.0
        if user.age < 25:
            score += 25.0   # Très adapté aux très jeunes
        if user.bank_status == 'basic':
            score += 10.0   # Parfait pour démarrer sans effort
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Épargnez passivement à chaque paiement par carte bancaire."]
        if user.age < 25:
            reasons.append("Parfaitement adapté aux étudiants et jeunes actifs pour épargner de petites sommes.")
        return reasons


# ==============================================================================
# SOUS-SECTION 4.2 : PRODUITS D'ASSURANCE (catégorie "insurance")
# ==============================================================================

class HealthInsurance(FinancialProduct):
    """Complémentaire santé – score renforcé avec l'âge et l'objectif urgence."""
    def __init__(self):
        super().__init__(
            "Complémentaire Santé PowerProtect",
            "insurance",
            "Couverture médicale complète pour vous et votre famille contre les imprévus de santé."
        )

    def recommend_score(self, user: User) -> float:
        """
        Score de base 70 ; augmente avec l'âge (risque santé croissant)
        et si l'objectif est la constitution d'un fonds d'urgence.
        """
        score = 70.0
        if user.age > 45:
            score += 20.0   # Risque santé plus élevé après 45 ans
        elif user.age > 30:
            score += 10.0   # Risque modéré entre 30 et 45 ans
        if user.financial_goal == 'emergency':
            score += 10.0   # Cohérent avec la volonté de se protéger
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Couverture médicale essentielle pour protéger votre capital santé."]
        if user.age > 45:
            reasons.append("Garanties renforcées et examens de prévention inclus adaptés à votre profil.")
        return reasons


class LifeInsurance(FinancialProduct):
    """Assurance vie – véhicule d'épargne et de transmission à long terme."""
    def __init__(self):
        super().__init__(
            "Assurance Vie PowerPrévoyance",
            "insurance",
            "Contrat d'assurance sur la vie permettant de constituer un capital à long terme."
        )

    def is_eligible(self, user: User) -> bool:
        """Réservé aux clients d'au moins 25 ans (maturité financière supposée)."""
        return user.age >= 25

    def recommend_score(self, user: User) -> float:
        """
        Score de base 50 ; fort bonus si l'objectif est la retraite (+35)
        et si le client est premium (+15, accès aux fonds en unités de compte).
        """
        score = 50.0
        if user.financial_goal == 'retirement':
            score += 35.0
        if user.bank_status == 'premium':
            score += 15.0
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Fiscalité attractive pour la transmission ou la valorisation à long terme."]
        if user.financial_goal == 'retirement':
            reasons.append("Idéal pour anticiper et préparer sereinement vos revenus de retraite.")
        return reasons


class CarInsurance(FinancialProduct):
    """Assurance automobile tous risques – réservée aux revenus suffisants."""
    def __init__(self):
        super().__init__(
            "Assurance Auto PowerDrive",
            "insurance",
            "Assurance automobile tous risques avec assistance immédiate en cas de panne."
        )

    def is_eligible(self, user: User) -> bool:
        """
        Condition d'éligibilité : salaire mensuel supérieur à 150 000 FCFA
        (on suppose qu'un véhicule nécessite un revenu minimum pour être entretenu).
        """
        return user.salary > 150000

    def recommend_score(self, user: User) -> float:
        """Score de base 60 ; bonus pour profil de risque élevé et statut premium."""
        score = 60.0
        if user.risk_level == 'high':
            score += 15.0   # Profil de risque élevé → assurance all-risks pertinente
        if user.bank_status == 'premium':
            score += 10.0   # Clients premium ont souvent un véhicule de valeur
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Protection multirisque complète pour votre véhicule personnel."]
        if user.risk_level == 'high':
            reasons.append("Option franchise réduite conseillée au vu de votre profil de risque.")
        return reasons


# ==============================================================================
# SOUS-SECTION 4.3 : PRODUITS D'INVESTISSEMENT (catégorie "investment")
# ==============================================================================

class Stocks(FinancialProduct):
    """Portefeuille d'actions – haut rendement potentiel, haut risque."""
    def __init__(self):
        super().__init__(
            "Portefeuille Actions PowerEquity",
            "investment",
            "Investissement direct sur les marchés boursiers mondiaux pour un rendement optimisé."
        )

    def is_eligible(self, user: User) -> bool:
        """
        Accessible uniquement aux profils avec tolérance au risque
        moyenne ou haute (risque "low" incompatible avec les actions).
        """
        return user.risk_level in ['medium', 'high']

    def recommend_score(self, user: User) -> float:
        """
        Score de base 40 (instrument risqué) ;
        +45 pour profil 'high' (parfaitement adapté),
        +20 pour profil 'medium',
        +15 si l'objectif est l'investissement pur.
        """
        score = 40.0
        if user.risk_level == 'high':
            score += 45.0
        elif user.risk_level == 'medium':
            score += 20.0
        if user.financial_goal == 'investment':
            score += 15.0
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Placement dynamique à fort potentiel de croissance sur les marchés d'actions."]
        if user.risk_level == 'high':
            reasons.append("Alignement parfait avec votre appétence élevée pour le risque.")
        return reasons


class Bonds(FinancialProduct):
    """Obligations souveraines – revenu régulier, faible risque."""
    def __init__(self):
        super().__init__(
            "Obligations Souveraines PowerYield",
            "investment",
            "Placements en titres de dettes d'État offrant un coupon régulier garanti."
        )

    def recommend_score(self, user: User) -> float:
        """
        Score de base 55 ;
        +30 pour profil risque 'low' (obligation = placement sécurisé),
        +10 pour profil 'medium',
        +10 si objectif retraite ou urgence (stabilité souhaitée).
        """
        score = 55.0
        if user.risk_level == 'low':
            score += 30.0
        elif user.risk_level == 'medium':
            score += 10.0
        if user.financial_goal in ['retirement', 'emergency']:
            score += 10.0
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Revenus réguliers avec un risque de perte en capital extrêmement bas."]
        if user.risk_level == 'low':
            reasons.append("Sécurisation idéale face aux fluctuations du marché actions.")
        return reasons


class Portfolio(FinancialProduct):
    """Gestion sous mandat – service de gestion déléguée pour les clients premium."""
    def __init__(self):
        super().__init__(
            "Gestion Sous Mandat PowerWealth",
            "investment",
            "Gestion déléguée sur-mesure opérée par nos experts en gestion de patrimoine."
        )

    def is_eligible(self, user: User) -> bool:
        """Réservé aux statuts 'standard' et 'premium' (capital minimum requis)."""
        return user.bank_status in ['standard', 'premium']

    def recommend_score(self, user: User) -> float:
        """
        Score de base 45 ;
        +40 pour statut premium (service sur-mesure),
        +15 si l'épargne dépasse 200 000 FCFA (capital intéressant à gérer).
        """
        score = 45.0
        if user.bank_status == 'premium':
            score += 40.0
        if user.savings > 200000:
            score += 15.0
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Pilotage d'actifs diversifié par nos gérants professionnels selon vos objectifs."]
        if user.bank_status == 'premium':
            reasons.append("Accès privilégié aux placements alternatifs (Private Equity, immobilier non coté).")
        return reasons


# ==============================================================================
# SOUS-SECTION 4.4 : PRODUITS DE CRÉDIT (catégorie "loans")
# ==============================================================================

class StudentLoan(FinancialProduct):
    """Crédit étudiant – taux préférentiel pour les jeunes en formation."""
    def __init__(self):
        super().__init__(
            "Crédit Étudiant PowerStart",
            "loans",
            "Prêt pour financer les études supérieures ou formations avec taux d'intérêt préférentiel."
        )

    def is_eligible(self, user: User) -> bool:
        """
        Conditions cumulatives :
        - Âge ≤ 28 ans (profil étudiant / jeune actif)
        - Statut 'basic' (pas encore de revenus stables)
        """
        return user.age <= 28 and user.bank_status == 'basic'

    def recommend_score(self, user: User) -> float:
        """Score de base 70 ; +25 pour les très jeunes (≤ 23 ans)."""
        score = 70.0
        if user.age <= 23:
            score += 25.0
        if user.financial_goal == 'investment':
            score += 5.0  # Investir dans sa formation = forme d'investissement
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Taux d'intérêt bonifié étudiant et différé de remboursement après l'obtention du diplôme."]
        if user.age <= 23:
            reasons.append("Recommandé pour lancer votre carrière universitaire ou de formation.")
        return reasons


class Mortgage(FinancialProduct):
    """Prêt immobilier – financement d'un achat de résidence principale."""
    def __init__(self):
        super().__init__(
            "Prêt Immobilier PowerHome",
            "loans",
            "Crédit immobilier amortissable à long terme pour l'achat de votre résidence principale."
        )

    def is_eligible(self, user: User) -> bool:
        """
        Trois conditions cumulatives pour l'éligibilité :
        - Âge ≥ 21 ans (majorité financière)
        - Salaire ≥ 200 000 FCFA (capacité de remboursement)
        - Épargne ≥ 30 000 FCFA (apport personnel minimum)
        """
        return user.age >= 21 and user.salary >= 200000 and user.savings >= 30000

    def recommend_score(self, user: User) -> float:
        """
        Score de base 45 ;
        +45 si l'objectif est 'housing' (achat immobilier explicitement souhaité),
        +10 pour clients premium (taux négociés).
        """
        score = 45.0
        if user.financial_goal == 'housing':
            score += 45.0
        if user.bank_status == 'premium':
            score += 10.0
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Financement immobilier sur-mesure avec des durées adaptables de 10 à 25 ans."]
        if user.financial_goal == 'housing':
            reasons.append("Spécifiquement conçu pour matérialiser votre projet d'achat de logement.")
        return reasons


class PersonalLoan(FinancialProduct):
    """Prêt personnel – crédit libre d'utilisation pour les revenus moyens."""
    def __init__(self):
        super().__init__(
            "Prêt Personnel PowerFlex-Loan",
            "loans",
            "Crédit à la consommation non affecté pour réaliser tous vos projets personnels."
        )

    def is_eligible(self, user: User) -> bool:
        """Accessible à partir d'un salaire de 130 000 FCFA (capacité de remboursement minimale)."""
        return user.salary >= 130000

    def recommend_score(self, user: User) -> float:
        """Score de base 50 ; +15 pour l'objectif urgence, +10 pour le risque moyen."""
        score = 50.0
        if user.financial_goal == 'emergency':
            score += 15.0   # Solution rapide en cas d'imprévu
        if user.risk_level == 'medium':
            score += 10.0
        return min(score, 100.0)

    def get_reason(self, user: User) -> list:
        reasons = ["Déblocage rapide des fonds sans justificatif de dépenses nécessaires."]
        if user.financial_goal == 'emergency':
            reasons.append("Solution de trésorerie d'appoint en cas d'imprévus financiers majeurs.")
        return reasons


# ==============================================================================
# ÉTAPE 5 : CATALOGUE DE PRODUITS – Fonction centrale de récupération
# ==============================================================================
def get_catalog():
    """
    Instancie et retourne la liste complète de tous les produits financiers
    disponibles dans l'offre PowerBank.
    Cette liste est passée au moteur de recommandation (Recommender.recommend()).
    """
    return [
        # Produits bancaires
        SavingsAccount(), CurrentAccount(), MicroSaving(),
        # Assurances
        HealthInsurance(), LifeInsurance(), CarInsurance(),
        # Investissements
        Stocks(), Bonds(), Portfolio(),
        # Crédits
        StudentLoan(), Mortgage(), PersonalLoan()
    ]


# ==============================================================================
# ÉTAPE 6 : GÉNÉRATION DE DONNÉES SYNTHÉTIQUES (NumPy + Faker)
# ==============================================================================
# Cette section crée une base de 100 clients fictifs mais statistiquement
# réalistes, simulant le portefeuille d'une banque digitale en Afrique de l'Ouest.

# Listes de prénoms et noms d'Afrique de l'Ouest pour un storytelling local fintech
WA_FIRST_NAMES = [
    'Koffi', 'Kouamé', 'Yao', 'Adjoua', 'Awa', 'Fatou', 'Moussa', 'Cheikh',
    'Amadou', 'Ousmane', 'Bakary', 'Mariam', 'Sékou', 'Diallo', 'Seydou',
    'Téné', 'Alassane', 'Tidiane', 'Soro', 'Gnamien', 'Bintou'
]
WA_LAST_NAMES = [
    'Kouadio', 'Koné', 'Diarrassouba', 'Traoré', 'Sylla', 'Coulibaly',
    'Diallo', 'Touré', 'Camara', 'Soro', 'Cissé', "N'Diaye", 'Bamba',
    'Ouattara', 'Gaye', 'Sow', 'Barry', 'Keïta', 'Sidibé'
]

def generate_synthetic_users(file_path, size=100):
    """
    Génère un fichier JSON contenant `size` profils clients fictifs.

    Utilise NumPy pour les distributions statistiques et Faker pour les noms.
    Injecte intentionnellement des anomalies pour illustrer le nettoyage Pandas.

    Paramètre :
        file_path (str) : Chemin du fichier JSON de sortie
        size      (int) : Nombre de profils à générer (défaut : 100)
    """
    # Instanciation de Faker en locale française (prénoms et noms francophones)
    fake = Faker('fr_FR')

    # Fixation des graines aléatoires pour la REPRODUCTIBILITÉ :
    # avec les mêmes graines, NumPy et random produiront toujours les mêmes valeurs.
    # Indispensable en hackathon pour que les résultats soient stables entre les runs.
    np.random.seed(42)
    random.seed(42)

    # --- Génération des âges (NumPy, distribution uniforme entière) ---
    # np.random.randint(low, high, size) : tire `size` entiers dans [low, high[
    ages = np.random.randint(18, 66, size=size)

    # --- Génération des salaires (NumPy, distribution normale) ---
    # np.random.normal(loc, scale, size) : distribution gaussienne centrée sur
    # loc=380 000 FCFA avec écart-type scale=140 000 FCFA.
    salaries = np.random.normal(loc=380000, scale=140000, size=size)
    # np.clip() borne les valeurs entre 100 000 et 1 000 000 FCFA
    # pour éviter les valeurs aberrantes (salaires négatifs ou astronomiques).
    salaries = np.clip(salaries, 100000, 1000000)

    # --- Génération de l'épargne (corrélée au salaire + bruit aléatoire) ---
    # savings_ratio : ratio d'épargne entre 8% et 35% du salaire (uniforme)
    savings_ratio = np.random.uniform(0.08, 0.35, size=size)
    # L'épargne est = salaire × ratio + bruit gaussien centré à 0 (écart-type 15 000)
    savings = salaries * savings_ratio + np.random.normal(loc=0, scale=15000, size=size)
    # Clipping : l'épargne ne peut pas être inférieure à 2 000 ni supérieure à 450 000 FCFA
    savings = np.clip(savings, 2000, 450000)

    # --- Génération des objectifs financiers (tirage pondéré NumPy) ---
    # np.random.choice() choisit parmi les options avec les probabilités p
    goals = ['retirement', 'housing', 'emergency', 'investment']
    goal_probs = [0.25, 0.35, 0.20, 0.20]  # Les probabilités doivent sommer à 1.0
    user_goals = np.random.choice(goals, size=size, p=goal_probs)

    # Niveaux de risque disponibles (utilisés dans la boucle ci-dessous)
    risk_options = ['low', 'medium', 'high']

    # Pools d'anomalies (~40 % des lignes touchées au total)
    rng_anom = np.random.default_rng(42)
    anomaly_pools = {
        "salary_none": set(rng_anom.choice(size, int(size * 0.15), replace=False)),
        "age_none": set(rng_anom.choice(size, int(size * 0.12), replace=False)),
        "savings_none": set(rng_anom.choice(size, int(size * 0.10), replace=False)),
        "savings_neg": set(rng_anom.choice(size, int(size * 0.14), replace=False)),
        "goal_format": set(rng_anom.choice(size, int(size * 0.18), replace=False)),
        "goal_invalid": set(rng_anom.choice(size, int(size * 0.10), replace=False)),
        "salary_str": set(rng_anom.choice(size, int(size * 0.07), replace=False)),
        "age_str": set(rng_anom.choice(size, int(size * 0.06), replace=False)),
        "salary_extreme": set(rng_anom.choice(size, int(size * 0.05), replace=False)),
        "age_extreme": set(rng_anom.choice(size, int(size * 0.05), replace=False)),
        "name_empty": set(rng_anom.choice(size, int(size * 0.04), replace=False)),
        "risk_invalid": set(rng_anom.choice(size, int(size * 0.08), replace=False)),
    }

    users_data = []  # Liste qui accumulera tous les dictionnaires clients

    # --- Boucle de génération des profils individuels ---
    for i in range(size):
        # Mélange de noms : 60% d'Afrique de l'Ouest, 40% Faker français
        if random.random() < 0.6:
            # Combinaison prénom africain + nom africain
            name = f"{random.choice(WA_FIRST_NAMES)} {random.choice(WA_LAST_NAMES)}"
        else:
            # Faker génère un prénom + nom aléatoires en français
            name = f"{fake.first_name()} {fake.last_name()}"

        age = int(ages[i])
        salary = float(salaries[i])
        user_savings = float(savings[i])
        goal = user_goals[i]

        # --- Déduction de la tolérance au risque selon le profil ---
        # Règle métier : certaines combinaisons (objectif + âge) déterminent
        # directement le niveau de risque (règles déterministes).
        if goal == 'emergency':
            risk = 'low'              # Urgence → prudence maximale
        elif goal == 'retirement' and age > 48:
            risk = 'low'              # Proche retraite → sécurisation
        elif goal == 'investment' and age < 32:
            risk = 'high'             # Jeune investisseur → prise de risque
        else:
            # Cas général : tirage probabiliste pondéré
            # 45% low, 40% medium, 15% high
            risk = np.random.choice(risk_options, p=[0.45, 0.40, 0.15])

        # --- Attribution du statut bancaire selon salaire et épargne ---
        if salary > 550000 or user_savings > 180000:
            status = 'premium'
        elif salary > 320000 or user_savings > 70000:
            status = 'standard'
        else:
            status = 'basic'

        # =========================================================
        # INJECTION MASSIVE D'ANOMALIES (pédagogie : voir le nettoyage)
        # ~40 % des lignes ont au moins un problème volontaire.
        # =========================================================
        if i in anomaly_pools["salary_none"]:
            salary = None
        if i in anomaly_pools["age_none"]:
            age = None
        if i in anomaly_pools["savings_none"]:
            user_savings = None
        if i in anomaly_pools["savings_neg"]:
            user_savings = float(random.choice([-12500.0, -8750.5, -3200.0, -99999.0]))
        if i in anomaly_pools["goal_format"]:
            goal = random.choice([
                goal.upper() + "   ",
                f"  {goal}  ",
                goal.replace("e", "E") + "!!!",
            ])
        if i in anomaly_pools["goal_invalid"]:
            goal = random.choice(["N/A", "", "achat_voiture", "retraite", "UNKNOWN", 12345])
        if i in anomaly_pools["salary_str"]:
            salary = random.choice(["erreur_api", "450000 FCFA", "non_renseigne", ""])
        if i in anomaly_pools["age_str"]:
            age = random.choice(["inconnu", "45 ans", ""])
        if i in anomaly_pools["salary_extreme"]:
            salary = float(random.choice([-85000.0, 99999999.0, 0.0]))
        if i in anomaly_pools["age_extreme"]:
            age = int(random.choice([5, 999, -3, 150]))
        if i in anomaly_pools["name_empty"]:
            name = random.choice(["", "   ", None])
        if i in anomaly_pools["risk_invalid"]:
            risk = random.choice(["", "extreme", "LOW", "inconnu", None])

        # Ajout du profil à la liste sous forme de dictionnaire
        users_data.append({
            "name": name,
            "age": age,
            "salary": salary,
            "savings": user_savings,
            "financial_goal": goal,
            "risk_level": risk,
            "bank_status": status
        })

    # --- Écriture des données dans le fichier JSON ---
    # open() avec 'w' crée ou écrase le fichier ; encoding='utf-8' pour les accents
    # json.dump() sérialise la liste Python en texte JSON formaté (indent=4)
    # ensure_ascii=False préserve les caractères spéciaux (é, ô, ñ…)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(users_data, f, indent=4, ensure_ascii=False)


# ==============================================================================
# ÉTAPE 7 : NETTOYAGE ET INGESTION DES DONNÉES (Pandas Pipeline)
#           Correspond à la Partie 2.2 du bootcamp
# ==============================================================================

def load_raw_users(file_path):
    """Charge le fichier JSON brut sans transformation."""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    return pd.DataFrame(raw_data)


VALID_GOALS = {'retirement', 'housing', 'emergency', 'investment'}
VALID_RISKS = {'low', 'medium', 'high'}


def _count_malformed_goals(df):
    return int(df['financial_goal'].apply(
        lambda x: str(x).strip().lower() not in VALID_GOALS if x is not None else True
    ).sum())


def _detect_anomaly_mask(df):
    salary_num = pd.to_numeric(df['salary'], errors='coerce')
    age_num = pd.to_numeric(df['age'], errors='coerce')
    savings_num = pd.to_numeric(df['savings'], errors='coerce')
    bad_goal = df['financial_goal'].apply(
        lambda x: str(x).strip().lower() not in VALID_GOALS if x is not None else True
    )
    bad_risk = df['risk_level'].apply(
        lambda x: str(x).strip().lower() not in VALID_RISKS if x is not None else True
    )
    bad_name = df['name'].apply(lambda x: x is None or str(x).strip() == "")
    return (
        df['salary'].isna() | salary_num.isna() & df['salary'].notna()
        | df['age'].isna() | age_num.isna() & df['age'].notna()
        | df['savings'].isna() | savings_num.isna() & df['savings'].notna()
        | (savings_num < 0) | (salary_num < 0)
        | (age_num < 18) | (age_num > 100)
        | bad_goal | bad_risk | bad_name
    )


def _style_raw_dataframe(df, theme=None):
    """Met en évidence null, négatif, type erroné et chaînes mal formatées."""
    t = theme_palette(theme)

    def _cell(bg_fg):
        bg, fg = bg_fg
        return f'background-color: {bg}; color: {fg}'

    def highlight_row(row):
        styles = [''] * len(row)
        salary_num = pd.to_numeric(row.get('salary'), errors='coerce')
        savings_num = pd.to_numeric(row.get('savings'), errors='coerce')
        for j, col in enumerate(row.index):
            val = row[col]
            if val is None or (isinstance(val, float) and pd.isna(val)) or val == '':
                styles[j] = _cell(t['anomaly_null'])
            elif col == 'savings' and pd.notna(savings_num) and savings_num < 0:
                styles[j] = _cell(t['anomaly_neg'])
            elif col == 'salary' and pd.notna(salary_num) and salary_num < 0:
                styles[j] = _cell(t['anomaly_neg'])
            elif col == 'financial_goal':
                g = str(val).strip().lower() if val is not None else ''
                if g not in VALID_GOALS:
                    styles[j] = _cell(t['anomaly_goal'])
            elif col == 'risk_level':
                r = str(val).strip().lower() if val is not None else ''
                if r not in VALID_RISKS:
                    styles[j] = _cell(t['anomaly_risk'])
            elif col in ('salary', 'age', 'savings') and isinstance(val, str):
                styles[j] = _cell(t['anomaly_type'])
            elif col == 'name' and (val is None or str(val).strip() == ''):
                styles[j] = _cell(t['anomaly_null'])
        return styles

    return df.style.apply(highlight_row, axis=1)


def clean_and_prepare_data(file_path, return_report=False):
    """
    Charge les données brutes depuis le fichier JSON, effectue toutes les
    opérations de nettoyage nécessaires et retourne un DataFrame Pandas propre.

    Opérations effectuées :
        1. Nettoyage des chaînes de caractères (objectifs financiers)
        2. Imputation des âges manquants (par la médiane)
        3. Imputation des salaires manquants (par la moyenne)
        4. Correction des valeurs d'épargne négatives (clipping à 0)
        5. Recalcul du statut bancaire (règle uniforme)

    Paramètres :
        file_path (str) : Chemin du fichier JSON brut
        return_report (bool) : Si True, retourne aussi le DataFrame brut et le rapport

    Retour :
        pd.DataFrame ou tuple (df_clean, df_raw, stats)
    """
    df_raw = load_raw_users(file_path)
    df = df_raw.copy()

    savings_num_raw = pd.to_numeric(df_raw['savings'], errors='coerce')
    salary_num_raw = pd.to_numeric(df_raw['salary'], errors='coerce')
    anomaly_mask = _detect_anomaly_mask(df_raw)

    stats = {
        "rows_before": len(df_raw),
        "missing_salary": int(df_raw['salary'].isna().sum()),
        "missing_age": int(df_raw['age'].isna().sum()),
        "missing_savings": int(df_raw['savings'].isna().sum()),
        "negative_savings": int((savings_num_raw < 0).sum()),
        "malformed_goals": _count_malformed_goals(df_raw),
        "invalid_types": int(
            pd.to_numeric(df_raw['salary'], errors='coerce').isna().sum()
            - df_raw['salary'].isna().sum()
            + pd.to_numeric(df_raw['age'], errors='coerce').isna().sum()
            - df_raw['age'].isna().sum()
            + pd.to_numeric(df_raw['savings'], errors='coerce').isna().sum()
            - df_raw['savings'].isna().sum()
        ),
        "rows_with_anomalies": int(anomaly_mask.sum()),
        "steps": [],
    }

    # ---- NETTOYAGE 0 : Noms vides → Faker ----
    fake_clean = Faker('fr_FR')
    n_empty_names = int(df['name'].apply(lambda x: x is None or str(x).strip() == "").sum())
    df['name'] = df['name'].apply(
        lambda x: x if x is not None and str(x).strip() else fake_clean.name()
    )
    stats["steps"].append({
        "name": "Correction des noms vides",
        "detail": f"{n_empty_names} nom(s) vide(s) remplacé(s) par des identités Faker",
    })

    # ---- NETTOYAGE 1 : Conversion des types numériques ----
    n_salary_type = int(pd.to_numeric(df['salary'], errors='coerce').isna().sum() - df['salary'].isna().sum())
    n_age_type = int(pd.to_numeric(df['age'], errors='coerce').isna().sum() - df['age'].isna().sum())
    n_savings_type = int(pd.to_numeric(df['savings'], errors='coerce').isna().sum() - df['savings'].isna().sum())
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['savings'] = pd.to_numeric(df['savings'], errors='coerce')
    stats["steps"].append({
        "name": "Normalisation des types",
        "detail": (
            f"Valeurs non numériques converties en NaN — "
            f"salaire: {n_salary_type}, âge: {n_age_type}, épargne: {n_savings_type}"
        ),
    })

    # ---- NETTOYAGE 2 : Normalisation des chaînes "financial_goal" ----
    n_goals_fixed = stats["malformed_goals"]
    df['financial_goal'] = df['financial_goal'].apply(
        lambda x: str(x).strip().lower() if x is not None else 'emergency'
    )
    valid_goals = ['retirement', 'housing', 'emergency', 'investment']
    df['financial_goal'] = df['financial_goal'].apply(
        lambda x: x if x in valid_goals else 'emergency'
    )
    stats["steps"].append({
        "name": "Normalisation des objectifs financiers",
        "detail": f"{n_goals_fixed} valeur(s) mal formatée(s) corrigée(s) (strip, lower, fallback 'emergency')",
    })

    # ---- NETTOYAGE 3 : Filtrage des valeurs extrêmes ----
    n_age_out = int(((df['age'] < 18) | (df['age'] > 100)).sum())
    n_salary_out = int(((df['salary'] < 100000) | (df['salary'] > 1000000)).sum())
    df.loc[(df['age'] < 18) | (df['age'] > 100), 'age'] = np.nan
    df.loc[(df['salary'] < 100000) | (df['salary'] > 1000000), 'salary'] = np.nan
    stats["steps"].append({
        "name": "Filtrage des valeurs aberrantes",
        "detail": f"Âges hors 18–100: {n_age_out} | Salaires hors 100k–1M FCFA: {n_salary_out}",
    })

    # ---- NETTOYAGE 4 : Normalisation risk_level ----
    n_bad_risk = int(df['risk_level'].apply(
        lambda x: str(x).strip().lower() not in VALID_RISKS if x is not None else True
    ).sum())
    df['risk_level'] = df['risk_level'].apply(
        lambda x: str(x).strip().lower() if x is not None and str(x).strip().lower() in VALID_RISKS else 'medium'
    )
    stats["steps"].append({
        "name": "Normalisation du niveau de risque",
        "detail": f"{n_bad_risk} valeur(s) invalide(s) remplacée(s) par 'medium'",
    })

    # ---- NETTOYAGE 5 : Imputation de l'âge manquant par la MÉDIANE ----
    n_age_missing = int(df['age'].isna().sum())
    median_age = int(df['age'].median(skipna=True))
    df['age'] = df['age'].fillna(median_age).astype(int)
    stats["steps"].append({
        "name": "Imputation de l'âge",
        "detail": f"{n_age_missing} valeur(s) manquante(s) remplacée(s) par la médiane ({median_age} ans)",
    })

    # ---- NETTOYAGE 6 : Imputation du salaire manquant par la MOYENNE ----
    n_salary_missing = int(df['salary'].isna().sum())
    mean_salary = df['salary'].mean(skipna=True)
    df['salary'] = df['salary'].fillna(mean_salary).round(2)
    stats["steps"].append({
        "name": "Imputation du salaire",
        "detail": f"{n_salary_missing} valeur(s) manquante(s) remplacée(s) par la moyenne ({mean_salary:,.0f} FCFA)",
    })

    # ---- NETTOYAGE 7 : Imputation épargne + correction négatives ----
    n_savings_missing = int(df['savings'].isna().sum())
    median_savings = df['savings'].median(skipna=True)
    df['savings'] = df['savings'].fillna(median_savings)
    n_neg_savings = int((df['savings'] < 0).sum())
    df['savings'] = df['savings'].apply(lambda x: max(0.0, float(x)))
    stats["steps"].append({
        "name": "Imputation & correction épargne",
        "detail": (
            f"{n_savings_missing} manquante(s) imputée(s) (médiane {median_savings:,.0f} FCFA) | "
            f"{n_neg_savings} négative(s) ramenée(s) à 0"
        ),
    })

    # ---- NETTOYAGE 8 : Recalcul consolidé du statut bancaire ----
    def determine_status(row):
        salary = row['salary']
        savings = row['savings']
        if salary > 550000 or savings > 180000:
            return 'premium'
        elif salary > 320000 or savings > 70000:
            return 'standard'
        return 'basic'

    status_before = df['bank_status'].value_counts().to_dict()
    df['bank_status'] = df.apply(determine_status, axis=1)
    status_after = df['bank_status'].value_counts().to_dict()
    stats["steps"].append({
        "name": "Recalcul du statut bancaire",
        "detail": f"Segmentation recalculée après imputation (avant: {status_before} → après: {status_after})",
    })

    stats["rows_after"] = len(df)
    stats["anomaly_rows"] = df_raw[anomaly_mask].copy()

    if return_report:
        return df, df_raw, stats
    return df


# ==============================================================================
# ÉTAPE 8 : MOTEUR DE SIMILARITÉ DE PROFILS (SciPy – Partie 4.4)
# ==============================================================================

def compute_profile_similarities(active_user, df_cleaned, n_recommendations=3):
    """
    Calcule le score de similarité entre un client cible et tous les autres
    clients en utilisant la distance euclidienne sur des vecteurs normalisés.

    Algorithme :
        1. Extraire les caractéristiques numériques (âge, salaire, épargne)
        2. Normaliser par Z-score (μ=0, σ=1) pour rendre les dimensions comparables
        3. Calculer les distances avec scipy.spatial.distance.cdist
        4. Convertir la distance en score de similarité : 1 / (1 + distance)
        5. Trier et retourner les N clients les plus proches

    Paramètres :
        active_user       (Recommender) : Client dont on cherche les sosies
        df_cleaned        (pd.DataFrame): Base clients nettoyée
        n_recommendations (int)         : Nombre de profils similaires à retourner

    Retour :
        pd.DataFrame : Les N clients les plus similaires avec leur score
    """
    # Sélection des 3 variables numériques utilisées comme dimensions du vecteur
    features = ['age', 'salary', 'savings']
    df_features = df_cleaned[features].copy()

    # --- Normalisation Z-score ---
    # Z-score : (valeur - moyenne) / écart-type
    # Transforme chaque variable en une dimension centrée (moyenne=0) et réduite (σ=1)
    # → permet de comparer âge, salaire et épargne sur le même pied d'égalité
    means = df_features.mean()
    stds = df_features.std()
    stds[stds == 0] = 1.0   # Évite la division par zéro si une variable est constante

    df_norm = (df_features - means) / stds

    # --- Normalisation du vecteur du client actif ---
    # On crée un tableau NumPy avec ses 3 caractéristiques puis on applique
    # le même Z-score calculé sur la base entière.
    active_vector = np.array([active_user.age, active_user.salary, active_user.savings])
    active_norm = (active_vector - means.values) / stds.values

    # --- Calcul des distances euclidiennes avec SciPy ---
    # distance.cdist(A, B) calcule la distance entre chaque ligne de A et chaque ligne de B.
    # A = matrice de tous les clients (N lignes × 3 colonnes)
    # B = matrice du client actif (1 ligne × 3 colonnes) → [active_norm]
    # metric='euclidean' : distance euclidienne = racine carrée de la somme des carrés des écarts
    # .flatten() transforme le résultat en vecteur 1D (au lieu d'une matrice Nx1)
    distances = distance.cdist(df_norm.values, [active_norm], metric='euclidean').flatten()

    # --- Conversion distance → score de similarité ---
    # Formule : similarité = 1 / (1 + distance)
    # - Si distance = 0 (profils identiques) → similarité = 1.0 (100%)
    # - Plus la distance est grande, plus la similarité tend vers 0
    similarities = 1 / (1 + distances)

    # Ajout du score au DataFrame et tri décroissant
    df_result = df_cleaned.copy()
    df_result['similarity_score'] = similarities
    df_result = df_result.sort_values(by='similarity_score', ascending=False)

    # Exclusion du client actif lui-même (comparaison exacte sur le nom)
    df_result = df_result[df_result['name'] != active_user.name]

    # Retour des N profils les plus similaires
    return df_result.head(n_recommendations)


# ==============================================================================
# ÉTAPE 9 : GÉNÉRATION DU DASHBOARD (Matplotlib & Seaborn – Partie 5)
# ==============================================================================

def build_dashboard_figure(df, catalog, theme=None):
    """
    Construit un dashboard Matplotlib/Seaborn 2×2 aux couleurs de PowerBank.

    Les 4 graphiques :
        [0,0] Barplot   : Répartition des objectifs financiers clients
        [0,1] Histplot  : Distribution des salaires avec courbe de densité (KDE)
        [1,0] Heatmap   : Croisement Niveau de Risque × Objectif Financier
        [1,1] Barplot   : Top 5 des produits recommandés par segment bancaire

    Retour :
        matplotlib.figure.Figure
    """
    active_theme = theme or get_app_theme()
    t = theme_palette(active_theme)
    sns.set_theme(style="dark" if active_theme == "dark" else "white")
    plt.rcParams.update({
        'figure.facecolor': t['plot_paper'],
        'axes.facecolor': t['plot_bg'],
        'text.color': t['plot_font'],
        'axes.labelcolor': t['text_subtle'],
        'xtick.color': t['text_subtle'],
        'ytick.color': t['text_subtle'],
        'axes.titlecolor': t['plot_title'],
        'grid.color': t['plot_grid'],
        'font.size': 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(t['plot_paper'])

    # ---- GRAPHIQUE 1 [0,0] : Barplot des objectifs financiers ----
    goal_counts = df['financial_goal'].value_counts()
    sns.barplot(
        x=goal_counts.index,    # Noms des objectifs sur l'axe X
        y=goal_counts.values,   # Nombre de clients par objectif sur l'axe Y
        palette='viridis',      # Palette de couleurs dégradée viridis
        ax=axes[0, 0]           # Axe cible dans la grille 2×2
    )
    axes[0, 0].set_title("1. Répartition des Objectifs Clients (Centres d'Intérêt)", fontsize=12, pad=10, weight='bold')
    axes[0, 0].set_xlabel("Objectif Financier")
    axes[0, 0].set_ylabel("Nombre de Clients")
    # Annotation des valeurs au-dessus de chaque barre
    for i, v in enumerate(goal_counts.values):
        axes[0, 0].text(i, v + 0.8, str(v), ha='center', color=t['plot_font'], fontweight='bold')

    # ---- GRAPHIQUE 2 [0,1] : Histogramme + KDE des salaires ----
    # kde=True : trace une courbe de densité (Kernel Density Estimation) par-dessus l'histogramme
    sns.histplot(
        data=df,
        x='salary',
        kde=True,               # Courbe de densité superposée
        color='#58A6FF',        # Bleu PowerBank
        ax=axes[0, 1],
        bins=15                 # Nombre de barres de l'histogramme
    )
    mean_sal = df['salary'].mean()
    # Ligne verticale pointillée à la moyenne (repère visuel clé)
    axes[0, 1].axvline(mean_sal, color='#FF7B72', linestyle='--', linewidth=2,
                       label=f'Moyenne: {mean_sal:,.0f} FCFA')
    axes[0, 1].set_title("2. Distribution des Salaires Mensuels", fontsize=12, pad=10, weight='bold')
    axes[0, 1].set_xlabel("Salaire (FCFA)")
    axes[0, 1].set_ylabel("Fréquence")
    axes[0, 1].legend()

    # ---- GRAPHIQUE 3 [1,0] : Heatmap Risque vs Objectif ----
    # pd.crosstab() crée un tableau de contingence : compte les co-occurrences de deux variables
    contingency = pd.crosstab(df['risk_level'], df['financial_goal'])
    sns.heatmap(
        contingency,
        annot=True,        # Affiche les valeurs numériques dans chaque cellule
        cmap='rocket_r',   # Palette de couleurs inversée (clair = peu, foncé = beaucoup)
        fmt='d',           # Format entier pour les annotations
        cbar=True,         # Affiche la barre de couleur à droite
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("3. Profils Clients : Risque vs Objectif Financier", fontsize=12, pad=10, weight='bold')
    axes[1, 0].set_xlabel("Objectif Financier")
    axes[1, 0].set_ylabel("Niveau de Risque")

    # ---- GRAPHIQUE 4 [1,1] : Top 5 produits recommandés par segment ----
    df_recom = _get_recommendation_counts(df, catalog)

    if not df_recom.empty:
        # Filtrer uniquement les 5 produits les plus recommandés au global
        # groupby + sum → total par produit ; nlargest(5) → top 5 ; .index → noms
        top_5_prod = df_recom.groupby("Produit")["Volume"].sum().nlargest(5).index
        df_recom_filtered = df_recom[df_recom["Produit"].isin(top_5_prod)]

        sns.barplot(
            data=df_recom_filtered,
            x="Volume",
            y="Produit",
            hue="Statut",       # Couleur différente par segment bancaire
            palette='muted',    # Palette de couleurs atténuées
            ax=axes[1, 1]
        )
    else:
        # Cas de secours si aucune recommandation n'a été générée
        axes[1, 1].text(0.5, 0.5, "Pas de recommandation générée.", ha='center', va='center')

    axes[1, 1].set_title("4. Recommandations par Segment (Top 5 Produits)", fontsize=12, pad=10, weight='bold')
    axes[1, 1].set_xlabel("Nombre de Recommandations")
    axes[1, 1].set_ylabel("")
    axes[1, 1].legend(title="Segment")

    plt.tight_layout()
    return fig


def dashboard_figure_to_png_bytes(fig, dpi=200):
    """Exporte la figure Matplotlib en PNG (mémoire) pour téléchargement jury."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


def _get_recommendation_counts(df, catalog):
    """Compte les recommandations produit par segment bancaire."""
    recom_by_status = {'basic': {}, 'standard': {}, 'premium': {}}
    for _, row in df.iterrows():
        rec_obj = Recommender(
            row['name'], row['age'], row['salary'], row['savings'],
            row['financial_goal'], row['risk_level'], row['bank_status']
        )
        recommendations = rec_obj.recommend(catalog)
        status = row['bank_status']
        for category, items in recommendations.items():
            for item in items[:2]:
                p_name = item['product'].name
                recom_by_status[status][p_name] = recom_by_status[status].get(p_name, 0) + 1

    flat_recom = []
    for status, counts in recom_by_status.items():
        for p_name, count in counts.items():
            flat_recom.append({"Statut": status, "Produit": p_name, "Volume": count})
    return pd.DataFrame(flat_recom)


def _style_plotly_figure(fig, theme=None, height=380):
    """Applique le thème actif aux graphiques Plotly (texte lisible)."""
    t = theme_palette(theme)
    fig.update_layout(
        template=t['plotly_template'],
        paper_bgcolor=t['plot_paper'],
        plot_bgcolor=t['plot_bg'],
        font=dict(color=t['plot_font'], size=12),
        title_font=dict(color=t['plot_title'], size=14),
        height=height,
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(
            bgcolor=t['bg_card'],
            bordercolor=t['border_strong'],
            font=dict(color=t['plot_font']),
        ),
    )
    fig.update_xaxes(
        gridcolor=t['plot_grid'], linecolor=t['border'],
        tickfont=dict(color=t['plot_font']), title_font=dict(color=t['plot_font']),
    )
    fig.update_yaxes(
        gridcolor=t['plot_grid'], linecolor=t['border'],
        tickfont=dict(color=t['plot_font']), title_font=dict(color=t['plot_font']),
    )
    return fig


def build_plotly_charts(df, catalog, theme=None):
    """
    Construit 4 graphiques Plotly interactifs (zoom, survol, légende).
    Retourne une liste de figures prêtes pour st.plotly_chart().
    """
    import plotly.express as px
    import plotly.graph_objects as go

    charts = []

    # 1 — Objectifs financiers
    goal_counts = df['financial_goal'].value_counts().reset_index()
    goal_counts.columns = ['Objectif', 'Clients']
    fig1 = px.bar(
        goal_counts, x='Objectif', y='Clients', color='Objectif',
        title="1. Répartition des objectifs clients",
        color_discrete_sequence=px.colors.sequential.Viridis,
        text='Clients',
    )
    t = theme_palette(theme)
    fig1.update_traces(textposition='outside', textfont=dict(color=t['plot_font']))
    charts.append(_style_plotly_figure(fig1, theme))

    # 2 — Distribution des salaires (histogramme + densité)
    fig2 = px.histogram(
        df, x='salary', nbins=20, marginal='box',
        title="2. Distribution des salaires (FCFA)",
        color_discrete_sequence=['#58A6FF'],
        hover_data=['name', 'bank_status'],
    )
    mean_sal = df['salary'].mean()
    fig2.add_vline(
        x=mean_sal, line_dash='dash', line_color='#FF7B72',
        annotation_text=f"Moyenne: {mean_sal:,.0f}",
        annotation_font_color=t['plot_font'],
    )
    charts.append(_style_plotly_figure(fig2, theme))

    # 3 — Heatmap risque × objectif
    contingency = pd.crosstab(df['risk_level'], df['financial_goal'])
    fig3 = px.imshow(
        contingency.values,
        x=contingency.columns.tolist(),
        y=contingency.index.tolist(),
        text_auto=True,
        color_continuous_scale='reds',
        title="3. Profils : risque vs objectif financier",
        labels=dict(color="Effectif"),
    )
    fig3.update_traces(textfont=dict(color=t['plot_font']))
    charts.append(_style_plotly_figure(fig3, theme))

    # 4 — Top recommandations par segment
    df_recom = _get_recommendation_counts(df, catalog)
    if not df_recom.empty:
        top_5 = df_recom.groupby("Produit")["Volume"].sum().nlargest(5).index
        df_top = df_recom[df_recom["Produit"].isin(top_5)]
        fig4 = px.bar(
            df_top, x='Volume', y='Produit', color='Statut',
            orientation='h', barmode='group',
            title="4. Top 5 produits recommandés par segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig4 = go.Figure()
        fig4.add_annotation(text="Pas de recommandation générée", showarrow=False,
                            font=dict(size=14, color=t['text_subtle']))
        fig4.update_layout(title="4. Top 5 produits recommandés par segment")
    charts.append(_style_plotly_figure(fig4, theme, height=420))

    return charts


def render_live_dashboard(df, catalog, theme=None):
    """Affiche le dashboard interactif Plotly dans Streamlit (pas d'image statique)."""
    charts = build_plotly_charts(df, catalog, theme)
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(charts[0], use_container_width=True, key="dash_chart_goals")
    with row1_col2:
        st.plotly_chart(charts[1], use_container_width=True, key="dash_chart_salary")
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(charts[2], use_container_width=True, key="dash_chart_heatmap")
    with row2_col2:
        st.plotly_chart(charts[3], use_container_width=True, key="dash_chart_recom")


# ==============================================================================
# ÉTAPE 10 : INITIALISATION AUTOMATIQUE DES FICHIERS AU DÉMARRAGE
# ==============================================================================
# Ces instructions s'exécutent une seule fois au premier lancement de l'app.
# Si les fichiers existent déjà, on ne les régénère pas (gain de temps).

# Génération / mise à jour des données avec anomalies pédagogiques
_needs_regen = not os.path.exists(USERS_JSON_PATH)
if os.path.exists(USERS_META_PATH):
    with open(USERS_META_PATH, encoding='utf-8') as _mf:
        _needs_regen = json.load(_mf).get("version") != USERS_DATA_VERSION
else:
    _needs_regen = True
if _needs_regen:
    generate_synthetic_users(USERS_JSON_PATH, size=100)
    with open(USERS_META_PATH, 'w', encoding='utf-8') as _mf:
        json.dump({"version": USERS_DATA_VERSION}, _mf)

# Chargement et nettoyage des données → DataFrame global disponible pour toutes les pages
df_cleaned, df_raw, cleaning_stats = clean_and_prepare_data(USERS_JSON_PATH, return_report=True)

# Instanciation du catalogue de produits (liste réutilisée dans toute l'application)
catalog = get_catalog()



# ==============================================================================
# ÉTAPE 11 : INTERFACE UTILISATEUR STREAMLIT (UI)
# ==============================================================================

# --- Thème sombre / clair (sidebar) ---
with st.sidebar:
    st.markdown("### ⚙️ Affichage")
    _light_on = st.toggle(
        "☀️ Mode clair",
        value=(st.session_state.get("app_theme", "dark") == "light"),
        help="Bascule entre thème sombre (défaut) et thème clair pour une meilleure lisibilité.",
    )
    st.session_state.app_theme = "light" if _light_on else "dark"
    st.markdown(f"<small>{'🌙 Sombre' if st.session_state.app_theme == 'dark' else '☀️ Clair actif'}</small>", unsafe_allow_html=True)

APP_THEME = get_app_theme()
inject_theme_css(APP_THEME)

# --- En-tête principale de l'application ---
st.title("🏦 Smart Finance Recommender – PowerAI_234")
st.caption("🏆 Solution Gagnante Hackathon COT_GenAI & Machine Learning 2026 — Simulation de conseiller intelligent PowerBank")

# --- Navigation par onglets (Tabs) ---
# st.tabs() crée un système de navigation à onglets horizontal.
# Retourne une liste d'objets "tab" utilisés comme gestionnaires de contexte (with tabs[i]:)
tabs = st.tabs([
    "🏛️ Accueil & Vue d'ensemble",
    "🧹 Nettoyage des Données",
    "👥 Base Clients PowerBank",
    "🎯 Conseiller Recommandations",
    "🧠 Explication du Moteur IA",
    "📊 Analyse Statistique (SciPy)",
    "📈 Dashboard Décisionnel (Jury)"
])


# ==============================================================================
# ONGLET 1 : ACCUEIL & VUE D'ENSEMBLE
# ==============================================================================
with tabs[0]:
    # Mise en page en 2 colonnes : grande colonne (2/3) + petite (1/3)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🏦 Smart Finance Recommender – PowerAI_234")
        st.write("""
        PowerBank est une banque digitale fictive dont le conseiller intelligent analyse les profils clients afin de recommander les solutions financières les plus adaptées : épargne, assurance, crédit et investissement.

        🎯 Objectif : illustrer l'utilisation de Python, de la Data Science et de la programmation orientée objet pour créer un système de recommandation financier explicable.
        """)

        st.subheader("⚙️ Architecture Fonctionnelle")
        # Diagramme ASCII dans un bloc de code pour une explication visuelle de l'architecture
        st.code("""
  +-----------------------------------------------------------+
  |              Génération des données clients               |
  |             (NumPy distributions + Faker API)             |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |                    data/users.json                        |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |     Onglet Nettoyage : filtrage & imputation (Pandas)     |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |   Pipeline ML optionnel (src/data_pipeline.py, 5 tranches)|
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |      Statistiques (SciPy Chi² & Normal Fit Profiler)      |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------+-----------------------------+
  |       Moteur de Recommandations OOP (Filter, Map, Reduce)  |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |   Dashboard interactif (Plotly + Matplotlib export)       |
  +-----------------------------------------------------------+
        """, language="text")

    with col2:
        st.subheader("📈 Indicateurs Globaux (KPIs)")

        # --- Calcul des indicateurs clés de performance ---
        total_users = len(df_cleaned)
        avg_sal = df_cleaned['salary'].mean()
        avg_sav = df_cleaned['savings'].mean()

        # st.metric() affiche une valeur chiffrée avec un label descriptif
        st.metric(label="Nombre de Clients PowerBank", value=f"{total_users}")
        st.write("")
        st.metric(label="Salaire Moyen Portefeuille", value=f"{avg_sal:,.0f} FCFA")
        st.write("")
        st.metric(label="Épargne Moyenne Portefeuille", value=f"{avg_sav:,.0f} FCFA")
        st.write("")

        # --- Test Chi² rapide affiché sur l'accueil ---
        # pd.crosstab() : tableau de contingence Statut vs Objectif
        contingency_table = pd.crosstab(df_cleaned['bank_status'], df_cleaned['financial_goal'])
        # stats.chi2_contingency() : test d'indépendance du Chi-deux
        # Retourne : statistique chi2, p-value, degrés de liberté, fréquences attendues
        chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)

        # Message textuel selon la significativité statistique (seuil 5%)
        status_text = "Relation Validée (Sig. < 5%)" if p_val < 0.05 else "Indépendance (p-value >= 5%)"
        st.metric(label="Test d'Hypothèse (Chi²)", value=f"{p_val:.4f}",
                  delta=status_text,
                  delta_color="normal" if p_val < 0.05 else "off")


# ==============================================================================
# ONGLET 2 : NETTOYAGE DES DONNÉES (Faker → JSON → Pandas Pipeline)
# ==============================================================================
with tabs[1]:
    st.subheader("🧹 Pipeline de filtrage et nettoyage des données")
    st.write(
        "Cet onglet montre le processus complet demandé en cours : génération **biaisée** "
        "avec **Faker + NumPy**, injection volontaire d'anomalies, puis nettoyage Pandas étape par étape."
    )

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Lignes brutes", cleaning_stats["rows_before"])
    col_kpi2.metric("Salaires manquants", cleaning_stats["missing_salary"])
    col_kpi3.metric("Âges manquants", cleaning_stats["missing_age"])
    col_kpi4.metric("Épargnes négatives", cleaning_stats["negative_savings"])

    col_kpi5, col_kpi6, col_kpi7, col_kpi8 = st.columns(4)
    col_kpi5.metric("Objectifs mal formatés", cleaning_stats["malformed_goals"])
    col_kpi6.metric("Types invalides", cleaning_stats.get("invalid_types", 0))
    col_kpi7.metric("Épargnes manquantes", cleaning_stats.get("missing_savings", 0))
    col_kpi8.metric("Lignes avec anomalies", cleaning_stats.get("rows_with_anomalies", len(cleaning_stats["anomaly_rows"])))

    st.caption(
        "Légende couleurs (données brutes) : "
        "🔴 rouge = manquant | 🟡 orange = négatif | 🟣 violet = objectif invalide | "
        "🔵 bleu = type erroné (texte au lieu de nombre)"
    )

    st.markdown("##### ⚙️ Étapes du pipeline de nettoyage")
    for i, step in enumerate(cleaning_stats["steps"], start=1):
        with st.expander(f"Étape {i} — {step['name']}", expanded=(i <= 2)):
            st.write(step["detail"])

    st.markdown("---")
    col_raw, col_clean = st.columns(2)

    with col_raw:
        st.markdown("##### ⚠️ Données brutes (`data/users.json`)")
        st.caption("Valeurs manquantes, négatives ou mal formatées visibles avant traitement.")
        st.dataframe(
            _style_raw_dataframe(df_raw, APP_THEME),
            use_container_width=True,
            height=450,
        )

    with col_clean:
        st.markdown("##### ✅ Données nettoyées (prêtes pour l'analyse)")
        st.caption("Après normalisation, imputation et recalcul du statut bancaire.")
        st.dataframe(df_cleaned, use_container_width=True, height=350)

    st.markdown("##### 🔍 Lignes contenant des anomalies injectées")
    if len(cleaning_stats["anomaly_rows"]) > 0:
        st.dataframe(cleaning_stats["anomaly_rows"], use_container_width=True)
    else:
        st.info("Aucune anomalie détectée dans le jeu actuel.")

    st.markdown("---")
    st.markdown("##### 🏗️ Pipeline ML avancé (`src/data_pipeline.py`)")
    pipeline_raw = os.path.join(DATA_DIR, "raw_data.csv")
    pipeline_clean = os.path.join(DATA_DIR, "clean_data.csv")
    pipeline_stats = os.path.join(DATA_DIR, "clean_stats.json")

    if os.path.exists(pipeline_raw) and os.path.exists(pipeline_clean):
        df_pipe_raw = pd.read_csv(pipeline_raw)
        df_pipe_clean = pd.read_csv(pipeline_clean)
        st.success(
            f"Dataset prêt bancaire disponible : **{len(df_pipe_raw):,}** lignes brutes → "
            f"**{len(df_pipe_clean):,}** lignes nettoyées (Faker + NumPy, 5 tranches ML)."
        )
        if os.path.exists(pipeline_stats):
            with open(pipeline_stats, encoding="utf-8") as fh:
                pipe_stats = json.load(fh)
            st.json(pipe_stats)
        pc1, pc2 = st.columns(2)
        with pc1:
            st.dataframe(df_pipe_raw.head(15), use_container_width=True)
        with pc2:
            st.dataframe(df_pipe_clean.head(15), use_container_width=True)
    else:
        st.info(
            "Lancez `python src/data_pipeline.py` pour générer le dataset de 5 000 lignes "
            "(Faker, biais genre/revenu, NaN MCAR/MAR, outliers) et le rapport ML complet."
        )
        if st.button("🚀 Exécuter le pipeline ML (5 tranches)"):
            import subprocess
            with st.spinner("Pipeline en cours…"):
                result = subprocess.run(
                    ["python", os.path.join(BASE_DIR, "src", "data_pipeline.py")],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                st.success("Pipeline terminé ! Rechargez l'onglet pour voir les résultats.")
                st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                st.rerun()
            else:
                st.error("Erreur lors de l'exécution du pipeline.")
                st.code(result.stderr or result.stdout)


# ==============================================================================
# ONGLET 3 : BASE CLIENTS POWERBANK
# ==============================================================================
with tabs[2]:
    st.subheader("👥 Consultation interactive des clients PowerBank")
    st.write("Filtrez en temps réel la base des 100 clients de la banque digitale fictive pour inspecter leurs détails.")

    # --- Widgets de filtrage (4 colonnes) ---
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        # Zone de texte pour rechercher par nom (case-insensitive)
        search_query = st.text_input("Rechercher un client (Nom) :", "")
    with col_f2:
        # Liste déroulante pour le statut bancaire
        status_filter = st.selectbox("Statut Bancaire :", ["Tous", "basic", "standard", "premium"])
    with col_f3:
        # Liste déroulante pour l'objectif financier
        goal_filter = st.selectbox("Objectif Financier :", ["Tous", "emergency", "housing", "retirement", "investment"])
    with col_f4:
        # Curseur double pour la tranche d'âge (min, max)
        min_age, max_age = st.slider("Tranche d'âge :", 18, 65, (18, 65))

    # --- Application des filtres Pandas en cascade ---
    df_filtered = df_cleaned.copy()
    if search_query:
        # str.contains() : filtre les lignes dont le nom contient la chaîne recherchée
        df_filtered = df_filtered[df_filtered['name'].str.contains(search_query, case=False)]
    if status_filter != "Tous":
        df_filtered = df_filtered[df_filtered['bank_status'] == status_filter]
    if goal_filter != "Tous":
        df_filtered = df_filtered[df_filtered['financial_goal'] == goal_filter]
    # Filtre composite sur l'âge (deux conditions avec l'opérateur & de Pandas)
    df_filtered = df_filtered[(df_filtered['age'] >= min_age) & (df_filtered['age'] <= max_age)]

    # Affichage du tableau interactif avec formatage des colonnes monétaires
    st.dataframe(
        df_filtered.style.format({
            "salary": "{:,.0f} FCFA",
            "savings": "{:,.0f} FCFA"
        }),
        use_container_width=True  # Utilise toute la largeur disponible
    )

    # --- Inspecteur de profil individuel ---
    st.subheader("🕵️ Inspecteur de Client individuel")
    selected_client_name = st.selectbox(
        "Sélectionnez un profil à auditer :",
        df_filtered['name'].tolist() if not df_filtered.empty else ["Aucun"]
    )

    if selected_client_name != "Aucun":
        # Récupération de la ligne du client sélectionné depuis le DataFrame complet
        client_data = df_cleaned[df_cleaned['name'] == selected_client_name].iloc[0]

        # --- Génération du badge HTML selon le statut ---
        # Le badge est rendu en HTML pur pour un style visuel personnalisé
        status = client_data['bank_status']
        if status == 'premium':
            badge_html = '<span class="badge-premium">Premium ✨</span>'
        elif status == 'standard':
            badge_html = '<span class="badge-standard">Standard 💳</span>'
        else:
            badge_html = '<span class="badge-basic">Basic ⚪</span>'

        # Affichage des informations en 3 colonnes
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            st.markdown(f"**Nom complet :** {client_data['name']}")
            st.markdown(f"**Âge :** {client_data['age']} ans")
            st.markdown(f"**Statut PowerBank :** {badge_html}", unsafe_allow_html=True)
        with c_col2:
            st.markdown(f"**Salaire mensuel :** {client_data['salary']:,.0f} FCFA")
            st.markdown(f"**Épargne totale :** {client_data['savings']:,.0f} FCFA")
        with c_col3:
            st.markdown(f"**Objectif principal :** `{client_data['financial_goal'].upper()}`")
            st.markdown(f"**Tolérance au risque :** `{client_data['risk_level'].upper()}`")

        # --- Analyse statistique personnalisée : centile de salaire (SciPy) ---
        sal_mean = df_cleaned['salary'].mean()
        sal_std = df_cleaned['salary'].std()
        # stats.norm.cdf() : Fonction de Répartition Cumulative (CDF) de la loi normale
        # Calcule P(X ≤ salaire_client) = la proportion de clients gagnant moins
        percentile = stats.norm.cdf(client_data['salary'], loc=sal_mean, scale=sal_std) * 100

        st.write("")
        # rf"..." : f-string brute permettant le LaTeX ($\mu$, $\sigma$) dans Streamlit Markdown
        st.markdown(rf"""
        > **🎯 Analyse statistique locale :** Le salaire de **{client_data['name']}** ({client_data['salary']:,.0f} FCFA) 
        > le place au centile **{percentile:.1f}%** (mieux rémunéré que {percentile:.1f}% de la population générale PowerBank). 
        > *(Calcul basé sur l'ajustement probabiliste SciPy $\mu = {sal_mean:,.0f}$, $\sigma = {sal_std:,.0f}$)*.
        """)


# ==============================================================================
# ONGLET 4 : CONSEILLER RECOMMANDATIONS
# ==============================================================================
with tabs[3]:
    st.subheader("🎯 Moteur de Recommandation Financière en Action")
    st.write("Sélectionnez un client dans la liste pour voir les propositions financières personnalisées générées par l'IA de PowerBank.")

    # Sélecteur de client (pré-sélectionne le client choisi dans l'onglet 2 si disponible)
    rec_client_name = st.selectbox(
        "Sélectionnez le client pour les recommandations :",
        df_cleaned['name'].tolist(),
        index=df_cleaned['name'].tolist().index(selected_client_name)
               if selected_client_name in df_cleaned['name'].tolist() else 0
    )

    # Récupération de la ligne du client sélectionné
    client_row = df_cleaned[df_cleaned['name'] == rec_client_name].iloc[0]

    # --- Instanciation du Recommender (héritage de User) ---
    # On crée un objet Recommender qui hérite de User et possède la méthode recommend()
    active_recommender = Recommender(
        name=client_row['name'],
        age=client_row['age'],
        salary=client_row['salary'],
        savings=client_row['savings'],
        financial_goal=client_row['financial_goal'],
        risk_level=client_row['risk_level'],
        bank_status=client_row['bank_status']
    )

    # --- Appel du moteur de recommandation (Filter → Map → Reduce) ---
    recoms = active_recommender.recommend(catalog)

    # Mise en page : recommandations à gauche (3/4) | clients similaires à droite (1/4)
    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.markdown(f"#### 🛍️ Offre de Recommandations pour : **{active_recommender.name}**")
        st.write(f"*Objectif principal :* `{active_recommender.financial_goal.upper()}` | *Risque :* `{active_recommender.risk_level.upper()}` | *Statut :* `{active_recommender.bank_status.upper()}`")

        # Mappage des clés de catégorie vers des libellés français avec émojis
        categories_fr = {
            "banking": "🏦 Banque & Comptes au Quotidien",
            "insurance": "🛡️ Assurances & Prévoyance",
            "investment": "📈 Placements & Investissements",
            "loans": "🏠 Solutions de Financement & Crédits"
        }

        # Itération sur les 4 catégories dans l'ordre souhaité
        for category_key, category_title in categories_fr.items():
            st.markdown(f"##### {category_title}")
            # recoms.get(key, []) : récupère la liste de recommandations pour cette catégorie
            # ou [] si aucun produit n'a été recommandé dans cette catégorie
            category_recoms = recoms.get(category_key, [])

            if not category_recoms:
                st.info("Aucun produit éligible pour cette catégorie au vu de votre profil.")
                continue  # Passe à la catégorie suivante

            for rec in category_recoms:
                prod = rec['product']
                score = rec['score']
                reasons = rec['reasons']

                # --- Codage couleur du score d'adéquation ---
                if score >= 80:
                    score_color = "#2ecc71"  # Vert vif → excellent match
                elif score >= 60:
                    score_color = "#f1c40f"  # Jaune → bon match
                else:
                    score_color = "#95a5a6"  # Gris → match modéré

                # Rendu de la carte produit en HTML personnalisé
                st.markdown(f"""
                <div class="recom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #58A6FF;">{prod.name}</h4>
                        <span style="background-color: {score_color}; color: #000000; padding: 3px 8px; border-radius: 5px; font-weight: bold; font-size: 13px;">
                            Indice Match : {score:.0f}%
                        </span>
                    </div>
                    <p style="font-style: italic; color: #8B949E; margin-top: 5px; font-size: 14px;">{prod.description}</p>
                    <div style="margin-top: 10px;">
                        <strong>🔍 Pourquoi ce choix ?</strong>
                        <ul style="margin-top: 5px; margin-bottom: 0; padding-left: 20px; font-size: 13px;">
                            {"".join([f"<li>{r}</li>" for r in reasons])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_r:
        st.markdown("#### 👥 Clients Similaires")
        st.write("Calculé via les distances Euclidiennes SciPy sur le profil financier (âge, salaire, épargne).")

        # Appel du moteur de similarité pour trouver les 3 clients les plus proches
        similar_clients = compute_profile_similarities(active_recommender, df_cleaned, n_recommendations=3)

        # Affichage de chaque client similaire sous forme de mini-carte HTML
        for idx, row in similar_clients.iterrows():
            sim_percent = row['similarity_score'] * 100  # Conversion en pourcentage
            st.markdown(f"""
            <div style="background-color: #161B22; border: 1px solid #21262D; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-weight: bold; color: #ffffff;">{row['name']}</div>
                <div style="font-size: 12px; color: #8B949E;">
                    Similarity : <b>{sim_percent:.1f}%</b><br/>
                    Age: {row['age']} ans | Statut: {row['bank_status'].upper()}<br/>
                    Goal: {row['financial_goal'].upper()}<br/>
                    Salary: {row['salary']:,.0f} FCFA
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# ONGLET 5 : EXPLICATION DU MOTEUR IA (pour le jury)
# ==============================================================================
with tabs[4]:
    st.subheader("🧠 Décryptage Technique du Moteur de Recommandation")
    st.write("Pour le jury du bootcamp, voici comment sont combinées la Programmation Orientée Objet (OOP) et la Programmation Fonctionnelle.")

    col_code1, col_code2 = st.columns(2)

    with col_code1:
        st.markdown("##### 1. Structure du Produit Financier (OOP Polymorphisme)")
        st.write("Chaque produit est instancié à partir d'une sous-classe de `FinancialProduct` qui surcharge l'éligibilité et le score.")
        # st.code() affiche le code avec coloration syntaxique
        st.code("""
# Classe mère
class FinancialProduct:
    def __init__(self, name, category, description):
        self.name = name
        self.category = category
        self.description = description

    def is_eligible(self, user) -> bool:
        return True

    def recommend_score(self, user) -> float:
        return 50.0

# Classe fille surchargeant le comportement (Polymorphisme)
class Mortgage(FinancialProduct):
    # Eligible si salaire >= 200 000 et apport (épargne) >= 30 000
    def is_eligible(self, user) -> bool:
        return user.age >= 21 and user.salary >= 200000 and user.savings >= 30000

    def recommend_score(self, user) -> float:
        score = 45.0
        if user.financial_goal == 'housing':
            score += 45.0
        if user.bank_status == 'premium':
            score += 10.0
        return min(score, 100.0)
        """, language="python")

    with col_code2:
        st.markdown("##### 2. Moteur IA Fonctionnel (Filter, Map, Reduce)")
        st.write("Le système utilise les fonctions de programmation fonctionnelle de Python natif avec `lambda` pour traiter le catalogue.")
        st.code("""
class Recommender(User):
    def recommend(self, catalog):
        # A. FILTER: Filtrer les produits éligibles
        eligible_products = list(
            filter(lambda p: p.is_eligible(self), catalog)
        )

        # B. MAP: Associer les scores et raisons à chaque produit éligible
        recommendations = list(
            map(lambda p: {
                "product": p,
                "score": p.recommend_score(self),
                "reasons": p.get_reason(self)
            }, eligible_products)
        )

        # Trier les offres par pertinence
        recommendations.sort(key=lambda r: r["score"], reverse=True)

        # C. REDUCE: Regrouper les recommandations par catégorie
        def group_by_category(accumulator, recommendation_item):
            category = recommendation_item["product"].category
            if category not in accumulator:
                accumulator[category] = []
            accumulator[category].append(recommendation_item)
            return accumulator

        return reduce(group_by_category, recommendations, {})
        """, language="python")

    st.success("💡 Ce design respecte les contraintes académiques les plus strictes : programmation fonctionnelle pure sans boucles explicitement imbriquées sur le traitement des listes de recommandations.")


# ==============================================================================
# ONGLET 6 : ANALYSE STATISTIQUE (SciPy)
# ==============================================================================
with tabs[5]:
    st.subheader("📊 Validation Mathématique des Préférences de Segment (SciPy)")
    st.write("L'une des étapes clés du bootcamp COT_GenAI est de valider scientifiquement les distributions de notre population via la statistique.")

    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.markdown("##### 📝 Statistiques Descriptives Globales")
        # .describe() génère automatiquement count, mean, std, min, quartiles, max
        st.dataframe(df_cleaned[['age', 'salary', 'savings']].describe(), use_container_width=True)

        st.markdown("##### 🏦 Distribution des statuts clients")
        # value_counts() : compte les occurrences de chaque valeur unique
        st.dataframe(df_cleaned['bank_status'].value_counts(), use_container_width=True)

    with col_stat2:
        # r"..." : raw string pour éviter l'interprétation des antislashs (LaTeX \chi)
        st.markdown(r"##### 🎲 Test d'Indépendance du Chi-Deux ($\chi^2$)")
        st.markdown("""
        **Hypothèse Nulle ($H_0$) :** L'objectif financier d'un client est indépendant de son statut bancaire (Basic, Standard, Premium).  
        **Hypothèse Alternative ($H_1$) :** Il existe une corrélation/association statistique entre l'objectif d'un client et son statut.
        """)

        # Tableau de contingence : croisement Statut × Objectif (calculé dans l'onglet 1)
        st.write("**Tableau de contingence observé (Croisement Statut vs Objectif) :**")
        st.dataframe(contingency_table)

        # --- Test Chi-deux (SciPy) ---
        # stats.chi2_contingency() calcule le test d'indépendance du Chi-deux sur un tableau de contingence
        # Retourne : chi2 (statistique), p_val (probabilité), dof (degrés de liberté), expected (effectifs attendus)
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

        # Affichage des métriques du test avec LaTeX Markdown
        st.markdown(rf"""
        - **Statistique de test $\chi^2$ :** `{chi2:.4f}`
        - **Degrés de liberté (dof) :** `{dof}`
        - **p-value calculée :** `{p_val:.6f}`
        """)

        # --- Interprétation de la p-value ---
        # Seuil classique : α = 5% (0.05)
        # Si p_val < 0.05 → on rejette H₀ (relation statistiquement significative)
        # Sinon → on ne peut pas rejeter H₀ (indépendance possible)
        if p_val < 0.05:
            st.markdown(f"""
            <div style="background-color: #1b4d22; border-left: 5px solid #2ecc71; padding: 12px; border-radius: 4px;">
                <b>Résultat du test : REJET DE L'HYPOTHÈSE NULLE (H₀)</b><br/>
                La p-value ({p_val:.6f}) est inférieure au seuil critique de 5%. Il existe une relation statistiquement significative
                entre la situation financière du client (Statut) et ses projets de vie (Objectif). 
                La segmentation algorithmique de PowerBank est donc mathématiquement justifiée.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #4a2b2b; border-left: 5px solid #e74c3c; padding: 12px; border-radius: 4px;">
                <b>Résultat du test : NON-REJET DE H₀</b><br/>
                La p-value ({p_val:.6f}) est supérieure ou égale à 5%. Il n'y a pas assez de preuves statistiques pour conclure à
                une dépendance entre l'objectif et le statut. Les clients partagent globalement les mêmes aspirations peu importe leur richesse.
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# ONGLET 7 : DASHBOARD DÉCISIONNEL (Plotly interactif + export Matplotlib)
# ==============================================================================
with tabs[6]:
    st.subheader("📈 Tableau de Bord Décisionnel")
    st.write(
        f"Graphiques **interactifs Plotly** recalculés en direct sur **{len(df_cleaned)}** clients "
        "(zoom, survol, légende). Matplotlib/Seaborn conservés pour l'export PNG jury."
    )

    render_live_dashboard(df_cleaned, catalog, APP_THEME)

    st.markdown("---")
    col_act1, col_act2, col_act3 = st.columns(3)

    with col_act1:
        st.write("💡 **Régénérer les données**")
        st.caption("Nouveau jeu Faker + NumPy → les 4 graphiques se mettent à jour immédiatement.")
        if st.button("🚀 Régénérer les données", key="regen_dashboard_data"):
            with st.spinner("Simulation en cours..."):
                generate_synthetic_users(USERS_JSON_PATH, size=100)
                with open(USERS_META_PATH, 'w', encoding='utf-8') as _mf:
                    json.dump({"version": USERS_DATA_VERSION}, _mf)
                st.success("Données régénérées — graphiques actualisés.")
                st.rerun()

    with col_act2:
        st.write("📥 **Exporter en PNG (Matplotlib)**")
        st.caption("Généré à la demande pour slides / rapport (pas affiché à l'écran).")
        _export_fig = build_dashboard_figure(df_cleaned, catalog, APP_THEME)
        _png_export = dashboard_figure_to_png_bytes(_export_fig)
        plt.close(_export_fig)
        st.download_button(
            label="🖼️ Télécharger dashboard.png",
            data=_png_export,
            file_name="powerbank_dashboard.png",
            mime="image/png",
            key="download_dashboard_png",
        )

    with col_act3:
        st.write("📥 **Exporter les données (CSV)**")
        st.caption("Profils clients nettoyés utilisés par les graphiques ci-dessus.")
        csv_data = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Télécharger profils CSV",
            data=csv_data,
            file_name="powerbank_cleaned_users.csv",
            mime="text/csv",
            key="download_dashboard_csv",
        )
