import torch
from translator import Encoder, Decoder, Seq2Seq
from preprocessing import (tokenize_en, add_special_tokens, build_vocab,
                           numericalize, en_word2idx, de_idx2word, de_word2idx)
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import spacy

spacy_en = spacy.load("en_core_web_sm")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def translate_sentence(sentence, model, en_word2idx, de_idx2word, max_len=50):
    model.eval()
    tokens = tokenize_en(sentence)
    tokens = ['<sos>'] + tokens + ['<eos>']
    numericalized = [en_word2idx.get(tok, en_word2idx['<unk>']) for tok in tokens]
    src_tensor = torch.tensor(numericalized, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    input_token = torch.tensor([de_word2idx['<sos>']], dtype=torch.long).to(DEVICE)
    output_sentence = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell)
        best_guess = output.argmax(1).item()
        if best_guess == de_word2idx['<eos>']:
            break
        output_sentence.append(de_idx2word[best_guess])
        input_token = torch.tensor([best_guess], dtype=torch.long).to(DEVICE)

    return output_sentence


INPUT_DIM = len(en_word2idx)
OUTPUT_DIM = len(de_word2idx)
EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
model.load_state_dict(torch.load("models/seq2seq_epoch_10.pt"))

sent = "how are you"
predicted = translate_sentence(sent, model, en_word2idx, de_idx2word)
print("EN:", sent)
print("PREDICTED DE:", " ".join(predicted))

reference = [['wie', 'geht', 'es', 'dir']]
candidate = predicted
bleu = sentence_bleu(reference, candidate)
print("BLEU score:", bleu)