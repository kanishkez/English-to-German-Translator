import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from translator import Encoder, Decoder, Seq2Seq
from preprocessing import load_and_preprocess, tokenize_en, tokenize_ger, TranslationDataset, collate_fn
from transformers import AutoTokenizer
from tqdm import tqdm
import os

os.makedirs("models", exist_ok=True)

english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
german_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

file_path = "translator/data/Sentence pairs in English-German - 2025-07-14.tsv"
df = load_and_preprocess(file_path)

tokenized_en = [tokenize_en(sent) for sent in tqdm(df['en'], desc="Tokenizing English")]
tokenized_ger = [tokenize_ger(sent) for sent in tqdm(df['ger'], desc="Tokenizing German")]

pad_idx = german_tokenizer.pad_token_id
train_dataset = TranslationDataset(tokenized_en, tokenized_ger)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_idx))

INPUT_DIM = english_tokenizer.vocab_size
OUTPUT_DIM = german_tokenizer.vocab_size
EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

for epoch in range(1, 11):
    loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), f"models/seq2seq_epoch_{epoch}.pt")
