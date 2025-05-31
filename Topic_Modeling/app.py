import streamlit as st
import pickle
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os

# Fonction pour charger le modèle
@st.cache_resource
def load_lda_model():
    try:
        with open('lda_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'lda_model.pkl' est introuvable. Veuillez vérifier son emplacement.")
        return None

# Fonction pour charger le DataFrame des avis
@st.cache_data
def load_reviews():
    try:
        df = pd.read_csv('reviews_with_topics.csv')
        return df
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'reviews_with_topics.csv' est introuvable.")
        return None

# Fonction de prétraitement du texte
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    return tokens

# Fonction pour prédire le topic
def predict_topic(new_text, lda_model, dictionary, topic_labels):
    processed_text = preprocess_text(new_text)
    if not processed_text:
        return None, 0, "Texte vide ou aucun mot valide après prétraitement"
    doc_bow = dictionary.doc2bow(processed_text)
    doc_topics = lda_model.get_document_topics(doc_bow)
    if doc_topics:
        dominant_topic = max(doc_topics, key=lambda x: x[1])
        return dominant_topic[0], dominant_topic[1], None
    return None, 0, "Aucun topic prédit"

# Interface Streamlit
st.title("Analyse de Topics - Avis Clients")

# Charger le modèle
model_data = load_lda_model()
if model_data is None:
    st.stop()

lda_model = model_data['lda_model']
dictionary = model_data['dictionary']
topic_labels = model_data['topic_labels']
num_topics = model_data['num_topics']

# Charger les avis
df_reviews = load_reviews()

# Afficher les topics identifiés
st.header("Topics Identifiés")
for idx, label in enumerate(topic_labels):
    st.write(f"**Topic {idx}**: {label}")

# Visualisation interactive avec pyLDAvis
st.header("Visualisation des Topics")
vis_file = 'lda_visualization.html'
if not os.path.exists(vis_file):
    vis = gensimvis.prepare(lda_model, model_data['corpus'], dictionary)
    pyLDAvis.save_html(vis, vis_file)

with open(vis_file, 'r') as f:
    html_string = f.read()
st.components.v1.html(html_string, height=800)

# Section pour analyser un nouvel avis
st.header("Analyser un Nouvel Avis")
user_input = st.text_area("Entrez un avis à analyser:", height=150)
if st.button("Analyser"):
    if user_input:
        topic_id, confidence, error = predict_topic(user_input, lda_model, dictionary, topic_labels)
        if error:
            st.error(error)
        elif topic_id is not None:
            st.success(f"**Topic identifié**: {topic_labels[topic_id]}")
            st.write(f"**Confiance**: {confidence:.3f}")
        else:
            st.error("Impossible de prédire le topic pour ce texte")
    else:
        st.warning("Veuillez entrer un texte à analyser.")

# Afficher des exemples d'avis
if df_reviews is not None:
    st.header("Exemples d'Avis du Jeu de Données")
    num_examples = st.slider("Nombre d'exemples à afficher:", 1, 10, 5)
    st.write(df_reviews[['text', 'dominant_topic', 'topic_label']].head(num_examples))
