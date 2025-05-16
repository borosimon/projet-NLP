import pandas as pd
import joblib # Import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Charger la matrice TF-IDF et le vecteur pour le modèle
tfidf_matrix = joblib.load('données vectorisées/global_tfidf_matrix.pkl') # Use joblib.load to load pickle file
tfidf_vectorizer = joblib.load('données vectorisées/global_tfidf_vectorizer.pkl')

# Assuming 'data' DataFrame still exists and contains 'label' column
# If not, you need to load it from the original data source

# Filtrer les lignes avec des labels non définis
data = pd.read_csv("modele d'entrainnement/global_posts_uvbf_annote.csv")
data = data.dropna(subset=['label'])

# Afficher le nombre de chaque label
print(data['label'].value_counts())


# Diviser les données
X = data['texte'].fillna('')  # Remplacez 'texte' par le nom de votre colonne de texte
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisation avec TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entraîner le modèle
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prédictions
y_pred = model.predict(X_test_tfidf)


# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Taux de commentaires
total_comments = len(data)
positive_comments = data[data['label'] == 'positif'].shape[0]
negative_comments = data[data['label'] == 'negatif'].shape[0]
neutral_comments = data[data['label'] == 'neutre'].shape[0]

positive_rate = positive_comments / total_comments * 100
negative_rate = negative_comments / total_comments * 100
neutral_rate = neutral_comments / total_comments * 100

print(f"Taux de commentaires positifs: {positive_rate:.2f}%")
print(f"Taux de commentaires négatifs: {negative_rate:.2f}%")
print(f"Taux de commentaires neutres: {neutral_rate:.2f}%")

