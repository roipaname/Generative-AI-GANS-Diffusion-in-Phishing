# =========================
# 1. SETUP FOR COLAB + GPU
# =========================
!pip install tiktoken joblib

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import tiktoken
import joblib
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("/content/drive/MyDrive/data/complaints_processed.csv")  # <-- Adjust path in Drive
df = df.drop_duplicates()
print(df.info())

tokenizer = tiktoken.get_encoding("cl100k_base")
label_encoder = LabelEncoder()
df = df.fillna("unknown")
df['product'] = label_encoder.fit_transform(df['product'])
max_length = 10

def encode_text_tiktoken(text):
    token_ids = tokenizer.encode(text, disallowed_special=())
    token_ids = token_ids[:max_length]  # truncate
    token_ids += [0] * (max_length - len(token_ids))  # pad
    return token_ids

df['encoded'] = df['narrative'].astype(str).apply(encode_text_tiktoken)

# =========================
# 3. DATASET + DATALOADER
# =========================
class ComplaintDataset(Dataset):
    def __init__(self, data):
        self.texts = torch.tensor(list(data['encoded']), dtype=torch.long)
        self.labels = torch.tensor(data['product'].values, dtype=torch.long)

    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

cdataset = ComplaintDataset(df)
dataloader = DataLoader(cdataset, shuffle=True, batch_size=32)

# =========================
# 4. MODEL
# =========================
class ComplaintClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=emb_dim)
        self.l1 = nn.Linear(emb_dim, 16)
        self.l2 = nn.Linear(16, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # average pooling
        x = torch.relu(self.l1(x))
        return self.l2(x)

VOCAB_SIZE = tokenizer.n_vocab
model = ComplaintClassifier(vocab_size=VOCAB_SIZE, emb_dim=64, num_classes=len(label_encoder.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================
# 5. TRAINING LOOP
# =========================
for epoch in range(50):  # Reduce to 10 epochs for quicker testing
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# =========================
# 6. SAVE TO DRIVE
# =========================
save_dir = "/content/drive/MyDrive/complaint_model"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f"{save_dir}/complaint_classifier.pth")
joblib.dump(label_encoder, f"{save_dir}/label_encoder.pkl")

print(f"Model and label encoder saved to: {save_dir}")

# =========================
# 7. LOAD MODEL FOR PREDICTION
# =========================
loaded_model = ComplaintClassifier(vocab_size=VOCAB_SIZE, emb_dim=64, num_classes=len(label_encoder.classes_))
loaded_model.load_state_dict(torch.load(f"{save_dir}/complaint_classifier.pth", map_location=device))
loaded_model.to(device).eval()

label_encoder = joblib.load(f"{save_dir}/label_encoder.pkl")

def predict(text):
    encoded = torch.tensor([encode_text_tiktoken(text)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = loaded_model(encoded)
        pred = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# Example usage
print(predict("The mortgage process was confusing and caused delays."))
