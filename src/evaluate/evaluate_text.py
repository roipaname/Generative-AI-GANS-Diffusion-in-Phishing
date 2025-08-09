from src.models.text.gpt_classifier import  GPTForClassification,GPTModel
import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import tiktoken
from src.dataloader.text_loader import EmailClassificationDataset
MODEL_PATH = "outputs/classification_model.pt"
CSV_PATH = ["data/phishing/nazario.csv"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
cfg = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
gpt_model = GPTModel(cfg)
model = GPTForClassification(gpt_model, hidden_size=cfg['emb_dim'], num_classes=2)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

dataset = EmailClassificationDataset(CSV_PATH, tokenizer)
dataloader = DataLoader(dataset, batch_size=8)
if __name__ == "__main__":
    with torch.no_grad():
        for input_ids, attention_mask, labels, bodies in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            for body, pred in zip(bodies, preds):
                print(f"Body:\n{body}\nPrediction: {'spam' if pred.item() == 1 else 'no spam'}\n{'-'*40}")