import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from translator import Encoder, Decoder, Seq2Seq
from preprocessing import (load_and_preprocess, tokenize_en, tokenize_ger,
                           add_special_tokens, build_vocab, numericalize,
                           TranslationDataset, collate_fn)


file_path = "translator/data/Sentence pairs in English-German - 2025-07-14.tsv"


df = load_and_preprocess(file_path)
tokenized_en = [tokenize_en(sent) for sent in df['en']]
tokenized_ger = [tokenize_ger(sent) for sent in df['ger']]
tokenized_en = add_special_tokens(tokenized_en)
tokenized_ger = add_special_tokens(tokenized_ger)


en_word2idx, en_idx2word = build_vocab(tokenized_en)
de_word2idx, de_idx2word = build_vocab(tokenized_ger)


numerical_en = numericalize(tokenized_en, en_word2idx)
numerical_ger = numericalize(tokenized_ger, de_word2idx)


pad_idx = de_word2idx['<pad>']
train_dataset = TranslationDataset(numerical_en, numerical_ger)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch, pad_idx))


INPUT_DIM = len(en_word2idx)
OUTPUT_DIM = len(de_word2idx)
EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)


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
    loss = train(model, train_loader, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), f"models/seq2seq_epoch_{epoch}.pt")
