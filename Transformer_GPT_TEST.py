# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

# === VOCABULAIRE TOY ===
token_to_id = {'<bos>': 0, '<eos>': 1, 'bonjour': 2, 'je': 3, 'suis': 4, 'un': 5, 'robot': 6}
id_to_token = {v: k for k, v in token_to_id.items()}
vocab_size = len(token_to_id)

# === GPT MINI-MODÈLE ===
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, ff_dim=128, num_layers=2, max_len=20, dropout=0.1):
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
def train(model, inputs, targets, optimizer, criterion, epochs=100):
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
def generate(model, start_tokens, max_len=10):
    model.eval()
    generated = start_tokens[:]
    for _ in range(max_len):
        input_tensor = torch.tensor([generated], dtype=torch.long)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.softmax(output[0, -1], dim=0)
        next_id = torch.multinomial(probs, num_samples=1).item()
        #next_id = torch.argmax(output[0, -1]).item()
        generated.append(next_id)
        if next_id == token_to_id['<eos>']:
            break
    return generated

def decode(token_ids):
    return ' '.join(id_to_token.get(i, '<unk>') for i in token_ids)

# === EXEMPLE TOY ===
sequence = ['<bos>', 'bonjour', 'je', 'suis', 'un', 'robot', '<eos>']
input_ids = torch.tensor([[token_to_id[t] for t in sequence[:-1]]])
target_ids = torch.tensor([[token_to_id[t] for t in sequence[1:]]])

model = MiniGPT(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Entraînement
train(model, input_ids, target_ids, optimizer, criterion)

# Génération
start = [token_to_id['<bos>']]
generated_ids = generate(model, start)
print("Texte généré :", decode(generated_ids))