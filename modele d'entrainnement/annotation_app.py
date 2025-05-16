
import streamlit as st
import pandas as pd

# Charger le dataset
# data = pd.read_csv('Texte nettoyé et prêt à être utilisé pour la vectorisation/posts_uvbf1_nettoye (1).csv')  # Remplacez par votre fichier
# data = pd.read_csv('/home/simonboro/Téléchargements/Projet NLP/Modèle de classification entraîné avec les données annotées/Texte nettoyé et prêt à être utilisé pour la vectorisation/posts_uvbf1_nettoye (1).csv')
data = pd.read_csv('données nettoyées/global_posts_uvbf_tokenizer.csv')

# Ajouter une colonne pour les labels
if 'label' not in data.columns:
    data['label'] = None

# Afficher le texte à annoter
for i in range(len(data)):
    st.write(data['texte'][i])  # Remplacez 'texte' par le nom de votre colonne
    label = st.radio("Choisissez un label :", options=['positif', 'neutre', 'négatif'], key=i)

    # Enregistrer le label
    data.at[i, 'label'] = label

# Sauvegarder le dataset annoté
if st.button('Sauvegarder les annotations'):
    data.to_csv('global_posts_uvbf_annote.csv', index=False)
    st.success("Annotations sauvegardées.")
