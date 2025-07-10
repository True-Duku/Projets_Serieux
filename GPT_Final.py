import torch
import torch.nn as nn
import numpy as np
from transformers import FlaubertTokenizer

# === TOKENIZER Flaubert ===
# Mémoire du PC insuffisante pour entrainer un tokenizeur sur un grand corpus
tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
vocab_size = tokenizer.vocab_size

# === GPT MINI-MODÈLE ===
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=50, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.positional_enc = self.create_positional_encoding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                div_term = 10000 ** (i / d_model)
                pe[pos, i] = np.sin(pos / div_term)
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.token_emb(input_ids) + self.positional_enc[:, :seq_len, :].to(input_ids.device)
        x = x.transpose(0, 1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        memory = torch.zeros(1, input_ids.size(0), x.size(-1), device=input_ids.device)
        decoded = self.decoder(x, memory, tgt_mask=mask)
        return self.out_proj(decoded.transpose(0, 1))

# === BOUCLE D'ENTRAÎNEMENT ===
def train(model, inputs, targets, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === GÉNÉRATION ===
def generate(model, start_text, max_len=20):
    model.eval()
    input_ids = tokenizer.encode(start_text, return_tensors="pt")
    generated = input_ids.tolist()[0]
    for _ in range(max_len):
        input_tensor = torch.tensor([generated], dtype=torch.long)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output[0, -1], dim=0)
            next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated)

# === TEST ===
model = MiniGPT(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# ======= NE FAIT PAS CE QUI EST DEMANDE ==============
# Text insuffisant pour entrainer le modèle => vrai corpus
#user_input = input("Pose-moi une question : ")
#text = user_input
#tokenized = tokenizer(text, return_tensors="pt")
#input_ids = tokenized["input_ids"]
#target_ids = input_ids.clone()
#target_ids[:, :-1] = input_ids[:, 1:]
#train(model, input_ids, target_ids, optimizer, criterion)
#variable = generate(model, text)
# print("Texte généré :",variable)

# ==== BLOCK DE CODE MODIFIE POUR UN VRAI ENTRAINEMENT ====
# === ENTRAÎNEMENT À PARTIR D'UN CORPUS TEXTE ===

# Charger le corpus depuis un fichier texte
with open("corpus.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Préparer les entrées et cibles tokenisées
inputs = []
targets = []

for line in lines:
    line = line.strip()
    if not line:
        continue  # ignorer les lignes vides
    tokenized = tokenizer(line, return_tensors="pt")["input_ids"]
    target = tokenized.clone()
    target[:, :-1] = tokenized[:, 1:]  # décalage pour prédiction auto-régressive

    inputs.append(tokenized)
    targets.append(target)

# Entraîner le modèle sur toutes les séquences du corpus
train(model, inputs, targets, optimizer, criterion)

# Utilisation après l'entraînement : génération à partir d'une question
user_input = input("Pose-moi une question : ")
response = generate(model, user_input)
print("Texte généré :", response)

