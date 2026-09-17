import os
import glob
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

# ---------------------------
# Dataset Definition with Pitch Normalization and Standardization
# ---------------------------
class ArticulatoryF0Dataset(Dataset):
    def __init__(self, artics_dir, f0_dir, target_mean=100.0):
        self.artics_files = sorted(glob.glob(os.path.join(artics_dir, "*.npy")))
        self.f0_files = sorted(glob.glob(os.path.join(f0_dir, "*.f0")))
        assert len(self.artics_files) == len(self.f0_files), "Mismatch between articulatory and f0 files"
        self.target_mean = target_mean
        self.global_mean = None
        self.global_std = None
        self.compute_f0_stats()

    def compute_f0_stats(self):
        """ Compute global mean and std for normalization """
        all_f0_values = []
        for f0_file in self.f0_files:
            f0 = self.f_read_raw_mat(f0_file, 1)
            non_zero = f0 != 0
            if np.any(non_zero):
                all_f0_values.append(f0[non_zero])
        all_f0_values = np.concatenate(all_f0_values)
        self.global_mean = all_f0_values.mean()
        self.global_std = all_f0_values.std()

    def __len__(self):
        return len(self.artics_files)

    def f_read_raw_mat(self, filename, col, data_format="f4", end="l"):
        f = open(filename, "rb")
        if end == "l":
            data_format = "<" + data_format
        datatype = np.dtype((data_format, (col,)))
        data = np.fromfile(f, dtype=datatype)
        f.close()
        if data.ndim == 2 and data.shape[1] == 1:
            return data[:, 0]
        else:
            return data

    def __getitem__(self, idx):
        artics = np.load(self.artics_files[idx])  # shape: (t, 64)
        f0 = self.f_read_raw_mat(self.f0_files[idx], 1)          # shape: (t,)

        # Interpolate f0 to match artics length if needed
        if len(f0) != artics.shape[0]:
            x_old = np.linspace(0, len(f0) - 1, num=len(f0))
            x_new = np.linspace(0, len(f0) - 1, num=artics.shape[0])
            f0 = np.interp(x_new, x_old, f0)

        # Normalize f0 to target mean (remove male/female bias)
        non_zero = f0 != 0
        if np.any(non_zero):
            current_mean = f0[non_zero].mean()
            scale = self.target_mean / current_mean
            f0[non_zero] = f0[non_zero] * scale

        # Standardize f0 (zero mean, unit variance)
        f0 = (f0 - self.global_mean) / self.global_std

        return torch.tensor(artics, dtype=torch.float32), torch.tensor(f0, dtype=torch.float32)

# ---------------------------
# Collate Function for Variable Length Sequences
# ---------------------------
def collate_fn(batch):
    artics, f0 = zip(*batch)  # Separate articulatory features and f0
    lengths = torch.tensor([a.shape[0] for a in artics], dtype=torch.long)
    artics_padded = pad_sequence(artics, batch_first=True, padding_value=0.0)
    f0_padded = pad_sequence(f0, batch_first=True, padding_value=0.0)
    return artics_padded, f0_padded, lengths

# ---------------------------
# Model Definition
# ---------------------------
class F0Predictor(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2):
        super(F0Predictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc(out).squeeze(-1)  # shape: (batch, max_t)
        return out

# ---------------------------
# Training & Validation Functions
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader)
    for i, (artics, f0, lengths) in enumerate(pbar):
        artics, f0, lengths = artics.to(device), f0.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(artics, lengths)
        loss = criterion(outputs, f0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * artics.size(0)
        pbar.set_description(f"Loss: {loss.item():.4f}")
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for artics, f0, lengths in dataloader:
            artics, f0, lengths = artics.to(device), f0.to(device), lengths.to(device)
            outputs = model(artics, lengths)
            loss = criterion(outputs, f0)
            running_loss += loss.item() * artics.size(0)
    return running_loss / len(dataloader.dataset)

# ---------------------------
# Main Training Script
# ---------------------------
def main():
    # Directories for data
    artics_dir = os.environ.get("ARTICS_DIR", "data/English/artics")
    f0_dir = os.environ.get("F0_DIR", "data/English/F0")
    
    # Hyperparameters
    batch_size = 128
    num_epochs = 1
    learning_rate = 1e-3
    valid_ratio = 0.1

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the dataset
    dataset = ArticulatoryF0Dataset(artics_dir, f0_dir, target_mean=100.0)
    
    # Split into training and validation sets
    total_samples = len(dataset)
    n_valid = int(total_samples * valid_ratio)
    n_train = total_samples - n_valid
    train_dataset, valid_dataset = random_split(dataset, [n_train, n_valid])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model, loss function, and optimizer
    model = F0Predictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = validate_epoch(model, valid_loader, criterion, device)
        print(f"Epoch [{epoch}/{num_epochs}]  Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}")
    
    # # Save trained model
    model_path = "f0_predictor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved as '{model_path}'.")

    # ---------------------------
    # Quick Inference (Single Sample)
    # ---------------------------
    print("\nPerforming Quick Inference...")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sample_artics, sample_f0, sample_lengths = next(iter(valid_loader))  # Get one batch
    sample_artics, sample_lengths = sample_artics.to(device), sample_lengths.to(device)

    with torch.no_grad():
        predicted_f0 = model(sample_artics, sample_lengths).squeeze().cpu().numpy()
    
    # Denormalize f0
    predicted_f0 = (predicted_f0 * dataset.global_std) + dataset.global_mean
    print(dataset.global_mean, dataset.global_std)
    print(f"Predicted f0 (first sample): {predicted_f0[0][:100]}")
    print(f"Actual f0 (first sample):    {sample_f0[0][:100].numpy() * dataset.global_std + dataset.global_mean}")

if __name__ == "__main__":
    main()
