# Analyse et Prédiction des Ventes de Smartphones 📱

Une application web interactive développée avec Streamlit pour analyser et prédire les prix des smartphones.

## 📋 Description

Cette application permet de :
- Visualiser les données de vente de smartphones
- Analyser les tendances des prix
- Prédire les prix de vente en fonction de différents critères

## 🚀 Fonctionnalités

### 1. Visualisation des données
- Distribution des prix
- Ventes par marque
- Relation prix/note
- Box plots des prix par marque
- Evolution des prix
- Corrélation prix-réduction
- Top 10 des modèles les plus chers
- Moyenne des réductions par marque

### 2. Prédiction des prix
- Sélection de la marque
- Choix de la mémoire RAM
- Choix du stockage
- Note utilisateur
- Prix original

## 🛠️ Technologies utilisées

- 🐍 Python 3.x
- 🌊 Streamlit
- 🐼 Pandas
- 📊 Plotly
- 🤖 Scikit-learn
- 📈 Matplotlib
- 🌟 Seaborn
- 📊 NumPy
- 📉 Statsmodels

## ⚙️ Installation

1. Clonez le repository :
```bash
git clone <url-du-repo>
cd Streamlit
```

2. Créez un environnement virtuel :
```bash
python -m venv env
env\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## 🚀 Lancement de l'application

```bash
streamlit run app.py
```

## 📊 Structure des données

Le fichier `Sales.csv` doit contenir les colonnes suivantes :
- Brands : Marque du smartphone
- Memory : Mémoire RAM
- Storage : Capacité de stockage
- Rating : Note utilisateur
- Original Price : Prix original
- Selling Price : Prix de vente
- discount percentage : Pourcentage de réduction

## 📝 Notes

- Les prix sont convertis de FCFA en USD pour une meilleure lisibilité
- Le modèle de prédiction utilise une régression linéaire
- Les visualisations sont interactives grâce à Plotly

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commit vos changements
4. Push sur la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.