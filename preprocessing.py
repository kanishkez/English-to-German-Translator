import torch
import pandas as pd
import spacy
import re
import string
from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\W', ' ', text)
    return text.strip()

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en(text)]

def tokenize_ger(text):
    return [tok.text.lower() for tok in spacy_de(text)]

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df.iloc[:,[1, 3]]
    df.columns = ['en', 'ger']
    df['en'] = df['en'].apply(clean_text)
    df['ger'] = df['ger'].apply(clean_text)
    return df

def add_special_tokens(sentences):
    return [['<sos>'] + sentence + ['<eos>'] for sentence in sentences]

def build_vocab(tokenized_sentences, min_freq=2):
    counter = Counter()
    for sentence in tokenized_sentences:
        counter.update(sentence)
    vocab = ['<pad>', '<unk>', '<sos>', '<eos>']
    vocab += [word for word, freq in counter.items() if freq >= min_freq]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def numericalize(sentences, word2idx):
    return [[word2idx.get(token, word2idx['<unk>']) for token in sentence] for sentence in sentences]

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