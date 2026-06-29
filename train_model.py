import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Constants
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
MAX_LEN = 1170
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DNADataset(Dataset):
    def __init__(self, sequences, labels, max_len=MAX_LEN):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx].upper()
        label = self.labels.iloc[idx]
        
        seq_encoded = np.full(self.max_len, 4, dtype=np.int64)
        for i, base in enumerate(seq):
            if i >= self.max_len:
                break
            seq_encoded[i] = BASE_TO_INT.get(base, 4)
            
        return torch.tensor(seq_encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

class DeepGenomicModel(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=16, hidden_dim=64):
        super(DeepGenomicModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=4)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=8)
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate the final forward and backward hidden states
        hidden = h_n.permute(1, 0, 2).reshape(-1, 128)
        
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1)

def main():
    parser = argparse.ArgumentParser(description="Train DNA Enhancer Predictor Model")
    parser.add_argument("--demo", action="store_true", help="Run a fast demo training on a small subset of the data")
    args = parser.parse_args()

    print("Loading Dataset...")
    df = pd.read_excel('CDD (17544 ones,2924 zeros).xlsx', header=1)
    
    if 'class' not in df.columns or 'sequence' not in df.columns:
        df.columns = ['class', 'sequence']
        
    print(f"Dataset shape: {df.shape}")
    
    if args.demo:
        print("Demo mode active: training on a tiny subset for quick validation.")
        # Take 1000 samples stratified by class
        df_0 = df[df['class'] == 0].sample(500, random_state=42)
        df_1 = df[df['class'] == 1].sample(500, random_state=42)
        df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)
        print(f"Demo Dataset shape: {df.shape}")
        epochs = 10
    else:
        epochs = EPOCHS

    X = df['sequence']
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = DNADataset(X_train, y_train)
    test_dataset = DNADataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight_val = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    
    print(f"Class imbalance handling: Pos weight = {pos_weight_val:.4f}")
    
    model = DeepGenomicModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch + 1} Completed | Train Loss: {avg_loss:.4f} ---")
        
        model.eval()
        all_targets = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()
                
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)
        
        print(f"Validation Metrics: Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    print("Training finished.")
    torch.save(model.state_dict(), 'enhancer_model.pth')
    print("Model saved to enhancer_model.pth")

if __name__ == '__main__':
    main()