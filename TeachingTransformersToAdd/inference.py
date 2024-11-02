import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, pad_idx):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def create_src_mask(self, src):
        src_mask = (src == self.pad_idx).transpose(0, 1)
        return src_mask

    def create_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt == self.pad_idx).transpose(0, 1)
        tgt_len = tgt.size(0)
        tgt_sub_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device) == 1, diagonal=1)
        tgt_sub_mask = tgt_sub_mask.bool()
        return tgt_sub_mask, tgt_pad_mask

    def forward(self, src, tgt):
        src_pad_mask = self.create_src_mask(src)
        tgt_mask, tgt_pad_mask = self.create_tgt_mask(tgt)

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer(src_emb, tgt_emb, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

def predict_sum(model, num1, num2, device, max_input_length, max_target_length, char2idx, idx2char):
    model.eval()
    input_str = f"{num1}+{num2}="
    tokens = [char2idx.get(c, 0) for c in input_str]
    tokens = tokens + [0] * (max_input_length - len(tokens))
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)

    decoder_input = torch.tensor([[char2idx['<s>']]], dtype=torch.long).to(device)

    result = ''
    with torch.no_grad():
        for _ in range(max_target_length):
            output = model(input_tensor, decoder_input)
            prediction = output.argmax(dim=-1)[-1, :]
            predicted_token = prediction.item()
            if predicted_token == char2idx['<eos>']:
                break
            predicted_char = idx2char.get(predicted_token, '')
            result += predicted_char
            decoder_input = torch.cat([decoder_input, prediction.unsqueeze(0)], dim=0)

    return result

# Model parameters (must match training parameters)
vocab_size = 16  # Adjusted to match training (0-15 indices)
d_model = 64     # Adjusted to match training
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 256
dropout = 0.1
pad_idx = 0
max_input_length = 13
max_target_length = 7  # Updated to match training

# Character to index mapping (must match training)
char2idx = {str(i): i+1 for i in range(10)}  # '0'-'9' -> 1-10
char2idx.update({'+': 11, '=': 12, ' ': 13})
char2idx['<s>'] = 14  # SOS token
char2idx['<eos>'] = 15  # EOS token
idx2char = {v: k for k, v in char2idx.items()}
idx2char[0] = ''  # PAD token

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Initialize model and load trained weights
model = TransformerModel(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    pad_idx=pad_idx
).to(device)

# Load the trained model
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Example usage
def calculate_sum(num1, num2):
    if not (0 <= num1 <= 9999 and 0 <= num2 <= 9999):
        return "Numbers must be between 0 and 9999"
    result = predict_sum(model, num1, num2, device, max_input_length, max_target_length, char2idx, idx2char)
    return result

# Interactive loop
if __name__ == "__main__":
    print("Addition Calculator (enter 'q' to quit)")
    print("Numbers should be between 0 and 9999")
    
    while True:
        try:
            num1 = input("\nEnter first number (or 'q' to quit): ")
            if num1.lower() == 'q':
                break
                
            num2 = input("Enter second number: ")
            if num2.lower() == 'q':
                break
                
            num1 = int(num1)
            num2 = int(num2)
            
            result = calculate_sum(num1, num2)
            print(f"\n{num1} + {num2} = {result}")
            print(f"Actual sum: {num1 + num2}")
            
        except ValueError:
            print("Please enter valid numbers")
        except Exception as e:
            print(f"An error occurred: {e}")