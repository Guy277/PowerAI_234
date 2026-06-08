# 🏦 PowerAI_234 — Smart Finance Recommender (PowerBank)

Simulation d'un conseiller bancaire intelligent pour la banque fictive **PowerBank**.

**Hackathon** — COT_GenAI & Machine Learning Bootcamp 2026

---

## 🚀 Lancement

```bash
pip install -r requirements.txt

# Pipeline complet (génération → analyse → reco → dashboard)
python main.py

# Interface Streamlit
streamlit run streamlit_app.py
```

---

## 📁 Structure

```
PowerAI_234/
├── main.py              # Fichier principal (tout le code)
├── streamlit_app.py     # Interface jury
├── data/
│   ├── users.json       # 100 clients simulés
│   └── dashboard.png    # Dashboard 2x2
└── requirements.txt
```

---

## 🧠 Compétences démontrées

- **POO** : encapsulation, getters, héritage (`Recommender`), polymorphisme
- **Python** : lambda, filter(), map(), reduce()
- **NumPy** : normal, clip, randint — données synthétiques
- **Pandas** : drop_duplicates, fillna, describe, value_counts
- **SciPy** : test Chi² + interprétation p-value
- **Matplotlib + Seaborn** : dashboard 2x2
