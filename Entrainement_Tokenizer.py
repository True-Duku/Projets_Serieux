# from tokenizers import BytePairEncoding
from tokenizers import ByteLevelBPETokenizer

from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer # Pour charger le tokenizer enregistré
import json



# Créer un fichier .txt à partir du JSON
with open("corpus_pypdf2.json", "r", encoding="utf-8") as f_in, open("temp_tokenizer_input.txt", "w", encoding="utf-8") as f_out:
    data = json.load(f_in)
    for doc in data:
        f_out.write(doc["text"].strip() + "\n\n")
		

files = ["temp_tokenizer_input.txt"] # Votre fichier extrait du PDF

# tokenizer = BytePairEncoding()
tokenizer = ByteLevelBPETokenizer()
tokenizer.pre_tokenizer = Whitespace()


print("Début de l'entraînement du tokenizer sur les PDFs...")
tokenizer.train(
    files,
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

print("Entraînement terminé.")

tokenizer.save("my_pdf_tokenizer.json")
print("Tokenizer basé sur les PDFs sauvegardé sous 'my_pdf_tokenizer.json'")

# Charger et tester
loaded_tokenizer = Tokenizer.from_file("my_pdf_tokenizer.json")
test_text_in_pdf = "Ceci est une phrase typique que l'on pourrait trouver dans mon document PDF."
test_text_out_of_pdf = "Le chat aime nager dans l'océan, un concept absurde."

print(f"\nTokenisation d'un texte IN des PDFs: {loaded_tokenizer.encode(test_text_in_pdf).tokens}")
print(f"Tokenisation d'un texte OUT des PDFs: {loaded_tokenizer.encode(test_text_out_of_pdf).tokens}")
