# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# === CONFIGURATION ===
class Config:
    def __init__(self):
        self.vocab_size = 10
        self.embedding_dim = 32
        self.num_heads = 2
        self.hidden_dim = 64
        self.num_layers = 2
        self.max_length = 10
        self.dropout_rate = 0.1

config = Config()

# === VOCABULAIRE TOY ===
token_to_id = {'<pad>': 0, '<bos>': 1, '<eos>': 2, 'je': 3, 'suis': 4, 'robot': 5, 'i': 6, 'am': 7, 'a': 8, 'robot_en': 9}
id_to_token = {v: k for k, v in token_to_id.items()}

# === DONNÉES TOY ===
src_sentence = ['je', 'suis', 'robot']
tgt_sentence = ['<bos>', 'i', 'am', 'a', 'robot_en', '<eos>']

src_tensor = torch.tensor([[token_to_id[t] for t in src_sentence]], dtype=torch.long)
tgt_tensor = torch.tensor([[token_to_id[t] for t in tgt_sentence]], dtype=torch.long)

class ToyDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return {'src': self.src[idx], 'tgt': self.tgt[idx]}

dataset = ToyDataset([src_tensor[0]], [tgt_tensor[0]])
data_loader = DataLoader(dataset, batch_size=1)

# === MODÈLE ===
class TransformerSeq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_src = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding_tgt = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = self.create_positional_encoding(config)

        encoder_layer = nn.TransformerEncoderLayer(config.embedding_dim, config.num_heads, config.hidden_dim, config.dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        decoder_layer = nn.TransformerDecoderLayer(config.embedding_dim, config.num_heads, config.hidden_dim, config.dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size)

    def create_positional_encoding(self, config):
        pos_enc = torch.zeros(config.max_length, config.embedding_dim)
        for pos in range(config.max_length):
            for i in range(0, config.embedding_dim, 2):
                div_term = 10000 ** (i / config.embedding_dim)
                pos_enc[pos, i] = np.sin(pos / div_term)
                if i + 1 < config.embedding_dim:
                    pos_enc[pos, i + 1] = np.cos(pos / div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, src, tgt):
        src_embed = self.embedding_src(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt_embed = self.embedding_tgt(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        src_embed = src_embed.transpose(0, 1)
        tgt_embed = tgt_embed.transpose(0, 1)
        memory = self.encoder(src_embed)
        output = self.decoder(tgt_embed, memory)
        return self.output_proj(output.transpose(0, 1))

# === ENTRAÎNEMENT ===
def train_model(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}")

# === GÉNÉRATION ===
def generate(model, src, tokenizer, max_len=10, start_token='<bos>', end_token='<eos>'):
    model.eval()
    device = src.device
    generated = [tokenizer[start_token]]
    for _ in range(max_len):
        tgt = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(src, tgt)
        next_id = torch.argmax(output[0, -1, :]).item()
        generated.append(next_id)
        if next_id == tokenizer[end_token]:
            break
    return generated

def decode_tokens(token_ids, id_to_token, end_token='<eos>'):
    tokens = []
    for idx in token_ids:
        token = id_to_token.get(idx, '<unk>')
        if token == end_token:
            break
        tokens.append(token)
    return ' '.join(tokens)

# === LANCEMENT ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerSeq2Seq(config).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, data_loader, criterion, optimizer, num_epochs=20, device=device)

generated_ids = generate(model, src_tensor.to(device), token_to_id)
translated_sentence = decode_tokens(generated_ids[1:], id_to_token)  # on ignore <bos>
print("Traduction générée :", translated_sentence)