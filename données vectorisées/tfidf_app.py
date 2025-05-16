from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd


data=pd.read_csv('données nettoyées/global_posts_uvbf_tokenizer.csv')
data['texte_pour_vectorisation'] = data['lemmes'].apply(lambda x: ' '.join(x))


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000)

# Create the TF-IDF matrix using fit_transform
tfidf_matrix = tfidf_vectorizer.fit_transform(data['texte_pour_vectorisation'])


# Sauvegarder la matrice TF-IDF et le vecteur pour le modèle
joblib.dump(tfidf_matrix, 'global_tfidf_matrix.pkl')
joblib.dump(tfidf_vectorizer, 'global_tfidf_vectorizer.pkl')


print("Dimensions de la matrice TF-IDF :", tfidf_matrix.shape)
print("Matrice de caractéristiques (TF-IDF) prête pour l'entraînement.")