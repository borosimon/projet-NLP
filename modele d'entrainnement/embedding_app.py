from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


from emoji import demojize

def process_emojis(text):
    return demojize(text)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

dataset=pd.read_csv('global_posts_uvbf_tokenizer_labeled.csv')


# !pip install sentence-transformers
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer


# Load the dataset (assuming it's already loaded as 'dataset')
# dataset = pd.read_csv('global_posts_uvbf_tokenizer_labeled.csv')

# Initialize a SentenceTransformer model for generating embeddings
model = SentenceTransformer('all-mpnet-base-v2') # Or any other suitable model

# Generate embeddings for the text data in the 'text' column
# Assuming your text data is in a column named 'text'
embeddings = model.encode(dataset['texte'].tolist())

# Now you can split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, dataset['label'], test_size=0.2)

# Proceed with SMOTE for oversampling
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = ((1 - pt) ** self.gamma) * BCE_loss
        return F_loss

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Exemple d'entraînement avec Logistic Regression
clf = LogisticRegression()
clf.fit(X_res, y_res)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label="positif")
plt.plot(fpr, tpr, marker='.')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

