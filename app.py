import streamlit as st 
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import LabelEncoder

# Style seaborn
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

# Convertit le prix de FCFA en USD
def fcfa_to_usd(fcfa):
    """Convertit le prix de FCFA en USD"""
    return fcfa / 583

# Chargement des données
@st.cache_data
def load_data(): 
    df = pd.read_csv("Sales.csv") 
    df.dropna(subset=['Selling Price', 'Original Price'], inplace=True)
    # Convertir les prix en USD
    df['Selling Price'] = df['Selling Price'].apply(fcfa_to_usd)
    df['Original Price'] = df['Original Price'].apply(fcfa_to_usd)
    return df

df = load_data()

st.title("Analyse et Prédiction des Ventes de Smartphones")

# Initialisation des encodeurs
brand_encoder = LabelEncoder()
memory_encoder = LabelEncoder()
storage_encoder = LabelEncoder()

# Fit une seule fois sur toutes les valeurs uniques connues
brands_unique = df['Brands'].dropna().unique()
memory_unique = df['Memory'].dropna().unique()
storage_unique = df['Storage'].dropna().unique()

brand_encoder.fit(brands_unique)
memory_encoder.fit(memory_unique)
storage_encoder.fit(storage_unique)

# Sidebar
option = st.sidebar.selectbox("Choisissez une action", [
    "Aperçu des données",
    "Visualisations",
    "Prédiction du prix"
])

# Aperçu
if option == "Aperçu des données":
    st.subheader("Aperçu des premières lignes")
    st.write(df.head())
    st.write("Dimensions:", df.shape)
    st.write("Colonnes:", df.columns.tolist())

# Visualisations
elif option == "Visualisations":
    st.subheader("Visualisation des données")

    vis_type = st.selectbox("Type de graphique", [
        "Distribution des prix",
        "Ventes par marque",
        "Prix vs Note",
        "Moyenne des réductions par marque",
        "Box Plot des prix par marque",  # Nouveau
        "Evolution des prix",  # Nouveau
        "Corrélation prix-réduction",  # Nouveau
        "Top 10 des modèles les plus chers"  # Nouveau
    ])

    if vis_type == "Distribution des prix":
        fig = px.histogram(df, x="Selling Price", 
                          title="Distribution des prix de vente (USD)",
                          labels={"Selling Price": "Prix (USD)"},
                          nbins=50)
        st.plotly_chart(fig)

    elif vis_type == "Ventes par marque":
        brand_counts = df["Brands"].value_counts().reset_index()
        brand_counts.columns = ["Marque", "Nombre"]
        fig = px.bar(brand_counts, x="Marque", y="Nombre",
                    title="Nombre de modèles par marque",
                    labels={"Marque": "Marque", "Nombre": "Nombre de modèles"})
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig)

    elif vis_type == "Prix vs Note":
        try:
            fig = px.scatter(df.dropna(subset=["Rating"]), 
                            x="Rating", y="Selling Price",
                            title="Relation entre Note et Prix de vente",
                            labels={"Rating": "Note", "Selling Price": "Prix (USD)"},
                            trendline="ols")
            st.plotly_chart(fig)
        except Exception as e:
            # Fallback sans ligne de tendance si statsmodels n'est pas disponible
            fig = px.scatter(df.dropna(subset=["Rating"]), 
                            x="Rating", y="Selling Price",
                            title="Relation entre Note et Prix de vente",
                            labels={"Rating": "Note", "Selling Price": "Prix (USD)"})
            st.plotly_chart(fig)
            st.warning("La ligne de tendance n'a pas pu être affichée. Installez statsmodels pour cette fonctionnalité.")

    elif vis_type == "Box Plot des prix par marque":
        fig = px.box(df, x="Brands", y="Selling Price",
                    title="Distribution des prix par marque",
                    labels={"Brands": "Marque", "Selling Price": "Prix (USD)"})
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig)

    elif vis_type == "Evolution des prix":
        # Calculer la moyenne mobile des prix
        df_sorted = df.sort_values("Original Price")
        fig = px.line(df_sorted, x="Original Price", y="Selling Price",
                     title="Evolution des prix de vente vs prix original",
                     labels={"Original Price": "Prix original (USD)", 
                            "Selling Price": "Prix de vente (USD)"})
        st.plotly_chart(fig)

    elif vis_type == "Corrélation prix-réduction":
        fig = px.scatter(df, x="Original Price", y="discount percentage",
                        title="Corrélation entre prix original et réduction",
                        labels={"Original Price": "Prix original (USD)", 
                               "discount percentage": "Réduction (%)"},
                        trendline="ols")
        st.plotly_chart(fig)

    elif vis_type == "Top 10 des modèles les plus chers":
        top_10 = df.nlargest(10, "Selling Price")
        fig = px.bar(top_10, x="Brands", y="Selling Price",
                    title="Top 10 des smartphones les plus chers",
                    labels={"Brands": "Marque", "Selling Price": "Prix (USD)"})
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig)

    elif vis_type == "Moyenne des réductions par marque":
        avg_discount = df.groupby("Brands")["discount percentage"].mean().reset_index()
        fig = px.bar(avg_discount, x="Brands", y="discount percentage",
                    title="Réduction moyenne (%) par marque",
                    labels={"Brands": "Marque", "discount percentage": "Réduction moyenne (%)"})
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig)

# Prédiction
elif option == "Prédiction du prix":
    st.subheader("Prédiction du prix de vente")

    df_model = df[['Brands', 'Memory', 'Storage', 'Rating', 'Original Price', 'Selling Price']].dropna()

    # Encodage
    df_model['Brands'] = brand_encoder.transform(df_model['Brands'])
    df_model['Memory'] = memory_encoder.transform(df_model['Memory'])
    df_model['Storage'] = storage_encoder.transform(df_model['Storage'])

    X = df_model.drop("Selling Price", axis=1)
    y = df_model["Selling Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Interface
    brand_input = st.selectbox("Marque", sorted(brands_unique))
    memory_input = st.selectbox("Mémoire RAM", sorted(memory_unique))
    storage_input = st.selectbox("Stockage", sorted(storage_unique))
    rating_input = st.slider("Note utilisateur", 0.0, 5.0, 4.0)
    original_price_input = st.number_input("Prix original (USD)", 
                                     min_value=2, 
                                     value=20, 
                                     step=1)

    if st.button("Prédire le prix"):
        try:
            input_df = pd.DataFrame({
                'Brands': [brand_encoder.transform([brand_input])[0]],
                'Memory': [memory_encoder.transform([memory_input])[0]],
                'Storage': [storage_encoder.transform([storage_input])[0]],
                'Rating': [rating_input],
                'Original Price': [original_price_input]
            })

            predicted_price = model.predict(input_df)[0]
            st.success(f"Prix prédit: {predicted_price:.2f} USD")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

