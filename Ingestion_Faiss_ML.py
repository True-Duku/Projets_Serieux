import PyPDF2
import re
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoModel, AutoTokenizer
import torch 
import faiss
#import pdfplumber
#import os

import numpy as np

# Ouvrir le fichier PDF en mode lecture binaire
with open("C:\\Users\\thier\\Desktop\Metabase_Dashboard\\Rag_Install_Diverse\\PDF_Stockage\\Fin_Du_Data_Analyste_.pdf", "rb") as fichier:
    # Créer un lecteur PDF
    lecteur = PyPDF2.PdfReader(fichier)
    
    # Initialiser une variable pour stocker le texte extrait
    texte_extrait = ""
    
    # Boucler à travers chaque page et extraire le texte
    for page_num in range(len(lecteur.pages)):
        page = lecteur.pages[page_num]
        texte_extrait += page.extract_text() + "\n"

    
# Afficher le texte extrait
#print(texte_extrait)
# nettoyage du texte
texte_brut = texte_extrait

def nettoyer_texte(texte):
   # Supprimer les espaces supplémentaires et les sauts de ligne
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

# Appliquer la fonction de nettoyage
texte_nettoye = nettoyer_texte(texte_brut)
#print(texte_nettoye)

# tokenization

# Texte à tokeniser
#texte_nettoye = "Voici un exemple de texte à découper en tokens."

# Créer une instance de TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)

# Découper le texte en morceaux (tokens)
tokens = splitter.split_text(texte_nettoye)

# Afficher les tokens
#print(tokens)

# Charger le modèle et le tokenizer
model_name = "sentence-transformers/labse"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fonction pour vectoriser le texte
def vectorize_text(text):
   inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
   with torch.no_grad():
       embeddings = model(**inputs).last_hidden_state.mean(dim=1)
   return embeddings

# Vectoriser le gros texte
vector = vectorize_text(tokens)

# Convertir le vecteur en un tableau NumPy (doit être de type float32)
vector_np = np.array(vector, dtype=np.float32).reshape(1, -1)  # Reshape pour ê
# Étape 1: Créer un index FAISS
dimension = vector_np.shape[1]  # Dimension de vos vecteurs
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance L2
# Étape 2: Ajouter le vecteur à l'index
index.add(vector_np)
# Étape 3: Sauvegarder l'index (optionnel)
#faiss.write_index(index, 'mon_index_faiss.index')
print("Vecteur ajouté à l'index avec succès.")


 

