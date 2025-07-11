# === CHARGER UN CORPUS MASSIF EN STREAMING ===
from tqdm import tqdm  # facultatif, juste pour une jolie barre de progression

def load_corpus(file_path, max_lines=10000):
    inputs, targets = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Chargement du corpus")):
            if i >= max_lines:
                break  # on limite à max_lines pour ne pas saturer la RAM
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer(line, return_tensors="pt")["input_ids"]
            target = tokens.clone()
            target[:, :-1] = tokens[:, 1:]  # prédiction du token suivant
            inputs.append(tokens)
            targets.append(target)
    return inputs, targets

# === UTILISATION ===
corpus_path = "fra_wikipedia_2021_Leipzig.txt"  # adapte ce nom si besoin
inputs, targets = load_corpus(corpus_path, max_lines=10000)  # tu peux ajuster max_lines

train(model, inputs, targets, optimizer, criterion)

# =================================================================================
#                   BOUCLE SUR PLUSIEURS FICHIERS
# =================================================================================
import os

corpus_folder = "fra_wikipedia_2021"  # remplace par ton nom de dossier
max_lines_per_file = 5000  # ajustable

for filename in os.listdir(corpus_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(corpus_folder, filename)
        print(f"\n Entraînement sur {filename}")
        inputs, targets = load_corpus(file_path, max_lines=max_lines_per_file)
        train(model, inputs, targets, optimizer, criterion)