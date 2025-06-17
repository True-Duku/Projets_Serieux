# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:18:33 2025

@author: thier
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import os


# Chargement du modèle ResNet pré-entraîné
resnet = models.resnet50(pretrained=True)
resnet.eval()  # Mode évaluation

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extraction des features d’une image
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Ajout d’une dimension batch
    with torch.no_grad():
        features = resnet(image)  # Extraction des features
    return features.numpy().flatten()

# Parcours du dossier et stockage des vecteurs
def process_images(folder_path, index_path="index_cards.faiss"):
    dimension = 2048  # Taille du vecteur ResNet
    index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance euclidienne

    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]

    for img_path in image_paths:
        vector = extract_features(img_path).astype("float32")
        index.add(np.array([vector]))  # Ajout du vecteur dans FAISS
        print(f"Vecteur enregistré pour {img_path}")

    #faiss.write_index(index, index_path)  # Sauvegarde de l’index
    print("Tous les vecteurs ont été sauvegardés !")
    
# Exécution : Choisissez le dossier à traiter
dossier = "C:\\Users\\thier\\Desktop\\Metabase_Dashboard\\image_ingestion\\train"
process_images(dossier)