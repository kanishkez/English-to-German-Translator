import torch
import pandas as pd
import re
import string
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
german_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

def tokenize_en(text):
    return english_tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)

def tokenize_ger(text):
    return german_tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df.iloc[:, [1, 3]]
    df.columns = ['en', 'ger']
    df['en'] = df['en'].apply(clean_text)
    df['ger'] = df['ger'].apply(clean_text)
    return df

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src = src_sequences
        self.trg = trg_sequences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "src": torch.tensor(self.src[idx], dtype=torch.long),
            "trg": torch.tensor(self.trg[idx], dtype=torch.long)
        }

def collate_fn(batch, pad_idx):
    src_batch = [item["src"] for item in batch]
    trg_batch = [item["trg"] for item in batch]
    src_padded = pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=pad_idx, batch_first=True)
    return src_padded, trg_padded
