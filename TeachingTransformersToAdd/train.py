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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        num1, num2, sum_result = self.data[idx]

        # Convert numbers to strings
        input_str = f"{num1}+{num2}="
        target_str = str(sum_result)

        # Tokenize and pad sequences
        input_tokens = self.tokenize_and_pad(input_str, self.max_input_length)

        # Create target tokens with <sos> and <eos>
        target_tokens = [self.sos_idx] + [self.char2idx.get(c, self.pad_idx) for c in target_str] + [self.eos_idx]

        # Pad target tokens
        target_tokens = target_tokens + [self.pad_idx] * (self.max_target_length - len(target_tokens))
        target_tokens = target_tokens[:self.max_target_length]

        # Assertions to check indices
        assert max(input_tokens) < vocab_size, f"Input contains invalid index {max(input_tokens)}"
        assert min(input_tokens) >= 0, f"Input contains negative index {min(input_tokens)}"
        assert max(target_tokens) < vocab_size, f"Target contains invalid index {max(target_tokens)}"
        assert min(target_tokens) >= 0, f"Target contains negative index {min(target_tokens)}"

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
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device).transpose(0, 1)  # Shape: [seq_len, batch_size]
        targets = targets.to(device).transpose(0, 1)  # Shape: [seq_len, batch_size]

        tgt_input = targets[:-1, :]
        tgt_output = targets[1:, :]

        optimizer.zero_grad()
        output = model(inputs, tgt_input)

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

def evaluate(model, dataloader, criterion, device, idx2char):
    model.eval()
    total_loss = 0
    total_seq_correct = 0
    total_token_correct = 0
    total_token_count = 0
    total_count = 0
    incorrect_examples = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device).transpose(0, 1)
            targets = targets.to(device).transpose(0, 1)

            batch_size = inputs.size(1)
            max_target_len = targets.size(0) - 1  # Exclude the initial SOS token

            # Initialize decoder input with SOS token
            decoder_input = torch.full((1, batch_size), train_dataset.sos_idx, dtype=torch.long, device=device)

            outputs = []
            batch_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for t in range(max_target_len):
                output = model(inputs, decoder_input)
                next_token = output.argmax(dim=-1)[-1, :]  # Shape: [batch_size]
                next_token[batch_finished] = train_dataset.pad_idx  # Ignore predictions for finished sequences
                outputs.append(next_token.unsqueeze(0))  # Shape: [1, batch_size]
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=0)
                batch_finished |= (next_token == train_dataset.eos_idx)
                if batch_finished.all():
                    break

            outputs = torch.cat(outputs, dim=0)  # Shape: [seq_len, batch_size]
            tgt_output = targets[1:, :]  # Exclude SOS token in target

            # Pad outputs to match tgt_output length
            if outputs.size(0) < tgt_output.size(0):
                pad_size = tgt_output.size(0) - outputs.size(0)
                outputs = torch.cat([outputs, torch.full((pad_size, batch_size), train_dataset.pad_idx, dtype=torch.long, device=device)], dim=0)
            else:
                outputs = outputs[:tgt_output.size(0), :]

            # Compute loss using teacher forcing for fair comparison
            tgt_input = targets[:-1, :]
            output_tf = model(inputs, tgt_input)
            loss = criterion(output_tf.view(-1, output_tf.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

            # Create mask to ignore positions after eos in both outputs and targets
            tgt_masks = []
            for i in range(batch_size):
                eos_positions = (tgt_output[:, i] == train_dataset.eos_idx).nonzero(as_tuple=False)
                if len(eos_positions) > 0:
                    eos_position = eos_positions[0].item()
                    mask = torch.zeros_like(tgt_output[:, i], dtype=torch.bool)
                    mask[:eos_position + 1] = True  # Include eos_idx
                else:
                    mask = torch.ones_like(tgt_output[:, i], dtype=torch.bool)
                tgt_masks.append(mask.unsqueeze(1))
            tgt_mask = torch.cat(tgt_masks, dim=1)  # Shape: [seq_len, batch_size]

            # Sequence-level accuracy
            seq_correct = (((outputs == tgt_output) | ~tgt_mask).all(dim=0)).sum().item()
            total_seq_correct += seq_correct

            # Token-level accuracy
            token_correct = ((outputs == tgt_output) & tgt_mask).sum().item()
            total_token_correct += token_correct
            total_token_count += tgt_mask.sum().item()

            total_count += batch_size

            # Collect incorrect examples
            for i in range(batch_size):
                predicted_seq = ''.join([idx2char.get(tok.item(), '') for tok in outputs[:, i] if tok.item() != train_dataset.pad_idx])
                target_seq = ''.join([idx2char.get(tok.item(), '') for tok in tgt_output[:, i] if tok.item() != train_dataset.pad_idx])
                if predicted_seq != target_seq and len(incorrect_examples) < 5:
                    incorrect_examples.append((predicted_seq, target_seq))

    avg_loss = total_loss / len(dataloader)
    seq_accuracy = total_seq_correct / total_count
    token_accuracy = total_token_correct / total_token_count

    # Print some incorrect examples
    if incorrect_examples:
        print("\nSample Incorrect Predictions:")
        for pred, tgt in incorrect_examples:
            print(f"Predicted: '{pred}', Target: '{tgt}'")
        print()

    return avg_loss, seq_accuracy, token_accuracy

def predict_sum(model, num1, num2, device, max_input_length, max_target_length, char2idx, idx2char):
    model.eval()
    input_str = f"{num1}+{num2}="
    tokens = [char2idx.get(c, 0) for c in input_str]
    tokens = tokens + [0] * (max_input_length - len(tokens))
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)  # Shape: [seq_len, 1]

    decoder_input = torch.tensor([[char2idx['<s>']]], dtype=torch.long).to(device)  # Start with SOS token

    result = ''
    with torch.no_grad():
        for i in range(max_target_length):
            output = model(input_tensor, decoder_input)
            prediction = output.argmax(dim=-1)[-1, :]
            predicted_token = prediction.item()
            if predicted_token == char2idx['<eos>']:
                break
            predicted_char = idx2char.get(predicted_token, '')
            result += predicted_char
            decoder_input = torch.cat([decoder_input, prediction.unsqueeze(0)], dim=0)

    return result

# %%
# 5. Hyperparameters and setup
# File paths for existing data
train_file = r"SyntheticData/addition_train.npy"
test_file = r"SyntheticData/addition_test.npy"
wild_ood_test_file = r"SyntheticData/addition_wild_ood_test.npy"

# Create datasets and dataloaders
max_input_length = 13  # Adjusted based on max possible input length
max_target_length = 7   # Adjusted based on max possible target length (5 digits + SOS + EOS)

train_dataset = AdditionDataset(train_file, max_input_length, max_target_length)
test_dataset = AdditionDataset(test_file, max_input_length, max_target_length)
wild_ood_test_dataset = AdditionDataset(wild_ood_test_file, max_input_length, max_target_length)

vocab_size = max(train_dataset.char2idx.values()) + 1  # Include padding index
pad_idx = train_dataset.pad_idx

batch_size = 128
num_epochs = 20  # Adjusted for testing
learning_rate = 5e-5  # Increased learning rate
d_model = 64
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 256
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
wild_ood_test_loader = DataLoader(wild_ood_test_dataset, batch_size=batch_size, drop_last=True)

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
wild_ood_losses = []
wild_ood_seq_accuracies = []
wild_ood_token_accuracies = []

best_seq_accuracy = 0
for epoch in range(num_epochs):
    avg_train_loss = train(model, train_loader, optimizer, criterion, device, epoch, writer)
    training_losses.append(avg_train_loss)

    val_loss, val_seq_accuracy, val_token_accuracy = evaluate(model, test_loader, criterion, device, train_dataset.idx2char)
    validation_losses.append(val_loss)
    validation_seq_accuracies.append(val_seq_accuracy)
    validation_token_accuracies.append(val_token_accuracy)

    wild_ood_loss, wild_ood_seq_accuracy, wild_ood_token_accuracy = evaluate(model, wild_ood_test_loader, criterion, device, train_dataset.idx2char)
    wild_ood_losses.append(wild_ood_loss)
    wild_ood_seq_accuracies.append(wild_ood_seq_accuracy)
    wild_ood_token_accuracies.append(wild_ood_token_accuracy)

    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Seq Accuracy: {val_seq_accuracy:.4f}, Token Accuracy: {val_token_accuracy:.4f}')
    print(f'Epoch {epoch+1}, Wild OOD Loss: {wild_ood_loss:.4f}, Seq Accuracy: {wild_ood_seq_accuracy:.4f}, Token Accuracy: {wild_ood_token_accuracy:.4f}')

    writer.add_scalar('Training Loss', avg_train_loss, epoch)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('Validation Sequence Accuracy', val_seq_accuracy, epoch)
    writer.add_scalar('Validation Token Accuracy', val_token_accuracy, epoch)
    writer.add_scalar('Wild OOD Loss', wild_ood_loss, epoch)
    writer.add_scalar('Wild OOD Sequence Accuracy', wild_ood_seq_accuracy, epoch)
    writer.add_scalar('Wild OOD Token Accuracy', wild_ood_token_accuracy, epoch)

    # Save the best model based on sequence accuracy on validation set
    if val_seq_accuracy > best_seq_accuracy:
        best_seq_accuracy = val_seq_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

    # Print an example prediction
    example_num1 = np.random.randint(0, 10000)
    example_num2 = np.random.randint(0, 10000)
    predicted_sum = predict_sum(model, example_num1, example_num2, device, max_input_length, max_target_length, train_dataset.char2idx, train_dataset.idx2char)
    print(f"Example Prediction: {example_num1} + {example_num2} = {predicted_sum}")

print(f'Best Validation Sequence Accuracy: {best_seq_accuracy:.4f}')
writer.close()

# Plot training, validation, and Wild OOD losses
plt.figure()
plt.plot(range(1, num_epochs+1), training_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), validation_losses, label='Validation Loss')
plt.plot(range(1, num_epochs+1), wild_ood_losses, label='Wild OOD Loss', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses')
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.close()

# Plot validation and Wild OOD accuracies
plt.figure()
plt.plot(range(1, num_epochs+1), validation_seq_accuracies, label='Validation Sequence Accuracy')
plt.plot(range(1, num_epochs+1), validation_token_accuracies, label='Validation Token Accuracy')
plt.plot(range(1, num_epochs+1), wild_ood_seq_accuracies, label='Wild OOD Sequence Accuracy', color='purple')
plt.plot(range(1, num_epochs+1), wild_ood_token_accuracies, label='Wild OOD Token Accuracy', color='pink')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation and Wild OOD Accuracies')
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
plt.close()

# Load the best model and make a prediction
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Test the model with an example
num1 = 1234
num2 = 5678
predicted_sum = predict_sum(model, num1, num2, device, max_input_length, max_target_length, train_dataset.char2idx, train_dataset.idx2char)
print(f"Final Prediction: {num1} + {num2} = {predicted_sum}")