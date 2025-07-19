import torch
from translator import Encoder, Decoder, Seq2Seq
from preprocessing import english_tokenizer, german_tokenizer
from nltk.translate.bleu_score import sentence_bleu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def translate_sentence(sentence, model, english_tokenizer, german_tokenizer, max_len=50):
    model.eval()
    token_ids = english_tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=128)
    src_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    input_token = torch.tensor([german_tokenizer.cls_token_id], dtype=torch.long).to(DEVICE)
    output_sentence = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell)
        top_token_id = output.argmax(1).item()
        if top_token_id == german_tokenizer.sep_token_id:
            break
        output_sentence.append(top_token_id)
        input_token = torch.tensor([top_token_id], dtype=torch.long).to(DEVICE)

    decoded_words = german_tokenizer.convert_ids_to_tokens(output_sentence)
    return decoded_words

INPUT_DIM = english_tokenizer.vocab_size
OUTPUT_DIM = german_tokenizer.vocab_size
EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
model.load_state_dict(torch.load("models/seq2seq_epoch_10.pt", map_location=DEVICE))

sentence = "how are you"
predicted = translate_sentence(sentence, model, english_tokenizer, german_tokenizer)
print("EN:", sentence)
print("PREDICTED DE:", " ".join(predicted))

reference = [['wie', 'geht', 'es', 'dir']]
candidate = predicted
bleu = sentence_bleu(reference, candidate)
print("BLEU score:", bleu)
