import argparse
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def download_and_load(record_name, dl_dir):
    print(f"Downloading record {record_name}...")
    wfdb.dl_database('mitdb', dl_dir=dl_dir, records=[record_name])
    record = wfdb.rdrecord(f'{dl_dir}/{record_name}')
    annotation = wfdb.rdann(f'{dl_dir}/{record_name}', 'atr')
    print("Record loaded.")
    return record, annotation

def extract_beats(signal, annotations, window=100):
    beats, labels = [], []
    for idx, sample in enumerate(annotations.sample):
        if sample - window >= 0 and sample + window < len(signal):
            beat = signal[sample - window: sample + window]
            beats.append(beat)
            labels.append(annotations.symbol[idx])
    return np.array(beats), np.array(labels)

def normalize_beats(beats):
    # Z-score normalization per beat
    return (beats - np.mean(beats, axis=1, keepdims=True)) / (np.std(beats, axis=1, keepdims=True) + 1e-8)

def encode_labels(labels):
    # Map all non-N to 1 (abnormal)
    label_map = {'N': 0}
    return np.array([label_map.get(label, 1) for label in labels])

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class ECGClassifier(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        conv_out_len = (input_len - 4) // 2  # after conv+pool
        self.fc1 = nn.Linear(16 * conv_out_len, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test Accuracy: {100 * acc:.2f}%")

def plot_beats(beats, labels):
    # MIT-BIH symbol to description mapping (add more as needed)
    symbol_map = {
        'N': 'Normal sinus',
        'L': 'Left bundle branch block',
        'R': 'Right bundle branch block',
        'A': 'Atrial premature',
        'V': 'Premature ventricular contraction',
        'F': 'Fusion of ventricular and normal',
        'J': 'Nodal (junctional) premature',
        'E': 'Atrial escape',
        'j': 'Nodal (junctional) escape',
        'a': 'Aberrated atrial premature',
        'S': 'Supraventricular premature',
        'e': 'Ventricular escape',
        'n': 'Supraventricular escape',
        '/': 'Paced',
        'f': 'Fusion of paced and normal',
        'Q': 'Unclassifiable',
        '?': 'Beat not classified',
        # Add more as needed
    }
    types_to_show = np.unique(labels)
    plt.figure(figsize=(12, 3 * len(types_to_show)))
    for i, label_type in enumerate(types_to_show):
        idxs = np.where(labels == label_type)[0]
        for j, idx in enumerate(idxs[:3]):  # Plot up to 3 beats per type
            plt.subplot(len(types_to_show), 3, i*3 + j + 1)
            plt.plot(beats[idx])
            beat_name = symbol_map.get(label_type, f"Unknown ('{label_type}')")
            plt.title(beat_name)
            plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=str, default='100', help='MIT-BIH record name')
    parser.add_argument('--window', type=int, default=100, help='Window size around R-peak')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dl_dir', type=str, default='mitdb_local', help='Download directory')
    args = parser.parse_args()

    record, annotation = download_and_load(args.record, args.dl_dir)
    signal = record.p_signal[:, 0]
    beats, labels = extract_beats(signal, annotation, window=args.window)
    beats = normalize_beats(beats)
    y = encode_labels(labels)

    print("Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"'{u}': {c} beats")

    X_train, X_test, y_train, y_test = train_test_split(beats, y, test_size=0.2, random_state=42)
    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGClassifier(input_len=beats.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, device, epochs=args.epochs)

    print("Evaluating...")
    evaluate_model(model, test_loader, device)

    plot_beats(beats, labels)

    # Save model
    torch.save(model.state_dict(), f"ecg_cnn_{args.record}.pt")
    print(f"Model saved as ecg_cnn_{args.record}.pt")

if __name__ == "__main__":
    main()