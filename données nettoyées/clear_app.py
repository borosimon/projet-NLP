import re
import emoji
import pandas as pd

import spacy
from spacy.lang.fr.stop_words import STOP_WORDS  # pour le français

data=pd.read_csv('global_posts_uvbf_labeled.csv')

#function de nettoyage 
def nettoyer_texte(texte):
    # Convertir l'entrée en chaîne de caractères pour gérer les valeurs non-chaînes
    texte = str(texte)
    # Supprimer les mentions (@user)
    texte = re.sub(r'@\w+', '', texte)
    # Supprimer les hashtags
    texte = re.sub(r'#\w+', '', texte)
    # Supprimer les URL
    texte = re.sub(r'http\S+|www.\S+', '', texte)
    # Supprimer la ponctuation
    texte = re.sub(r'[^\w\s]', '', texte)
    # Supprimer les emojis
    texte = emoji.replace_emoji(texte, replace="")
    # Transformer en minuscules
    texte = texte.lower()
    return texte

# Exemple d'application
data['texte_nettoye'] = data['texte'].apply(nettoyer_texte)

data.to_csv('global_posts_uvbf_nettoye.csv', index=False)


# Charger le modèle spaCy pour le français
nlp = spacy.load('fr_core_news_md')

#function tokenizer
def tokenizer_et_supp_stop_words(texte):
    doc = nlp(texte)
    tokens = [token.text for token in doc if token.text not in STOP_WORDS and not token.is_punct]
    return tokens

data['tokens'] = data['texte_nettoye'].apply(tokenizer_et_supp_stop_words)

#function lemmatizer
def lemmatizer(texte):
    doc = nlp(texte)
    lemmes = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return lemmes

data['lemmes'] = data['texte_nettoye'].apply(lemmatizer)



data.to_csv('global_posts_uvbf_tokenizer.csv', index=False)
