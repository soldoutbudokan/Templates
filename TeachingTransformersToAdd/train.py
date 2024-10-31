# %%
# Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt

# Set the working directory (optional, depending on your setup)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the output directory for plots
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)

# %%
# 2. Data Processing and Loading
class AdditionDataset(Dataset):
    def __init__(self, filename, max_input_length, max_target_length):
        self.data = np.load(filename, mmap_mode='r')
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Build the character to index mapping
        self.char2idx = {str(i): i+1 for i in range(10)}  # '0'-'9' -> 1-10
        self.char2idx.update({'+': 11, '=': 12, ' ': 13})
        self.pad_idx = 0
        self.sos_idx = max(self.char2idx.values()) + 1
        self.char2idx['<s>'] = self.sos_idx
        self.eos_idx = self.sos_idx + 1
        self.char2idx['<eos>'] = self.eos_idx
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.idx2char[self.pad_idx] = ''

        # Ensure pad_idx is not equal to eos_idx
        assert self.pad_idx != self.eos_idx, "pad_idx should not be equal to eos_idx"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        num1, num2, sum_result = self.data[idx]

        # Convert numbers to strings
        input_str = f"{num1}+{num2}="
        target_str = str(sum_result) + '<eos>'

        # Tokenize and pad sequences
        input_tokens = self.tokenize_and_pad(input_str, self.max_input_length)
        target_tokens = [self.sos_idx] + self.tokenize_and_pad(target_str, self.max_target_length - 1)

        # Ensure target_tokens has the correct length
        target_tokens = target_tokens + [self.pad_idx] * (self.max_target_length - len(target_tokens))
        target_tokens = target_tokens[:self.max_target_length]

        # Convert to tensors
        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)

        return input_tensor, target_tensor

    def tokenize_and_pad(self, s, max_length):
        tokens = [self.char2idx.get(c, self.pad_idx) for c in s]
        tokens = tokens + [self.pad_idx] * (max_length - len(tokens))
        return tokens[:max_length]  # Ensure the sequence is not longer than max_length

# %%
# 3. Model Architecture
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

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

# %%
# 4. Training and Evaluation Functions
def train(model, dataloader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0
    running_loss = 0
    teacher_forcing_ratio = max(0.5 * (0.99 ** epoch), 0.1)  # Example schedule

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device).transpose(0, 1)  # Shape: [seq_len, batch_size]
        targets = targets.to(device).transpose(0, 1)  # Shape: [seq_len, batch_size]

        tgt_input = targets[:-1, :]
        tgt_output = targets[1:, :]

        optimizer.zero_grad()

        use_teacher_forcing = True if np.random.rand() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            output = model(inputs, tgt_input)
        else:
            # Without teacher forcing
            decoder_input = tgt_input[:1, :]
            outputs = []
            for t in range(tgt_input.size(0)):
                out = model(inputs, decoder_input)
                top1 = out.argmax(2)[-1, :].unsqueeze(0)
                decoder_input = torch.cat([decoder_input, top1], dim=0)
                outputs.append(out[-1, :, :].unsqueeze(0))
            output = torch.cat(outputs, dim=0)

        loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()

        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = running_loss / 100
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.4f}')
            writer.add_scalar('Training Loss', avg_loss, epoch * len(dataloader) + batch_idx)
            running_loss = 0

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# %%
# 5. Hyperparameters and setup
# File paths for existing data
train_file = r"SyntheticData/addition_train.npy"
test_file = r"SyntheticData/addition_test.npy"

# Create datasets and dataloaders
max_input_length = 13  # Adjusted based on max possible input length
max_target_length = 7   # Adjusted based on max possible target length (+1 for SOS token, +1 for EOS token)

train_dataset = AdditionDataset(train_file, max_input_length, max_target_length)
test_dataset = AdditionDataset(test_file, max_input_length, max_target_length)

vocab_size = max(train_dataset.char2idx.values()) + 1  # Include padding index
pad_idx = train_dataset.pad_idx

batch_size = 128
num_epochs = 20
learning_rate = 1e-4
d_model = 256
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 1024
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

# %%
# 6. Initialize the model
model = TransformerModel(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, pad_idx=pad_idx).to(device)

# %%
# 7. Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0  # Set to zero for now
)


# %%
# 8. TensorBoard setup
writer = SummaryWriter('runs/addition_transformer')

# %%
# 9. Training loop and tests
training_losses = []
validation_losses = []
validation_seq_accuracies = []
validation_token_accuracies = []

best_seq_accuracy = 0
for epoch in range(num_epochs):
    avg_train_loss = train(model, train_loader, optimizer, criterion, device, epoch, writer)
    training_losses.append(avg_train_loss)

    val_loss, val_seq_accuracy, val_token_accuracy = evaluate(model, test_loader, criterion, device, train_dataset.idx2char)
    validation_losses.append(val_loss)
    validation_seq_accuracies.append(val_seq_accuracy)
    validation_token_accuracies.append(val_token_accuracy)

    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Seq Accuracy: {val_seq_accuracy:.4f}, Token Accuracy: {val_token_accuracy:.4f}')
    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('Validation Sequence Accuracy', val_seq_accuracy, epoch)
    writer.add_scalar('Validation Token Accuracy', val_token_accuracy, epoch)

    # Save the best model based on sequence accuracy
    if val_seq_accuracy > best_seq_accuracy:
        best_seq_accuracy = val_seq_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

    # Print an example prediction
    example_num1 = np.random.randint(0, 1000)
    example_num2 = np.random.randint(0, 1000)
    predicted_sum = predict_sum(model, example_num1, example_num2, device, max_input_length, max_target_length, train_dataset.char2idx, train_dataset.idx2char)
    print(f"Example Prediction: {example_num1} + {example_num2} = {predicted_sum}")

print(f'Best Validation Sequence Accuracy: {best_seq_accuracy:.4f}')
writer.close()

# Plot training and validation loss
plt.figure()
plt.plot(range(1, num_epochs+1), training_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.close()

# Plot validation accuracies
plt.figure()
plt.plot(range(1, num_epochs+1), validation_seq_accuracies, label='Sequence Accuracy')
plt.plot(range(1, num_epochs+1), validation_token_accuracies, label='Token Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracies')
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
plt.close()

# Load the best model and make a prediction
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Test the model with an example
num1 = 123
num2 = 678
predicted_sum = predict_sum(model, num1, num2, device, max_input_length, max_target_length, train_dataset.char2idx, train_dataset.idx2char)
print(f"Final Prediction: {num1} + {num2} = {predicted_sum}")