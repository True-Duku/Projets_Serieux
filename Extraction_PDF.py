import os
import json
import PyPDF2

# Dossier contenant les fichiers PDF
dossier_pdfs = "Documentation"

# Liste qui va contenir les textes extraits
documents = []

# Parcourir tous les fichiers PDF dans le dossier
for nom_fichier in os.listdir(dossier_pdfs):
    if nom_fichier.lower().endswith(".pdf"):
        chemin_pdf = os.path.join(dossier_pdfs, nom_fichier)

        try:
            with open(chemin_pdf, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                texte = ""
                for page in reader.pages:
                    texte += page.extract_text() or ""

            documents.append({
                "filename": nom_fichier,
                "text": texte.strip()
            })

        except Exception as e:
            print(f"Erreur lors du traitement de {nom_fichier} : {e}")

# Écriture dans un fichier JSON
with open("corpus_pypdf2.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"{len(documents)} fichiers PDF extraits avec PyPDF2 et enregistrés dans 'corpus_pypdf2.json'")