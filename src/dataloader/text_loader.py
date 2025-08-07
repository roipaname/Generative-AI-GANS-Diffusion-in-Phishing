
import glob
import torch, torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import tiktoken # type: ignore
import pandas as pd # type: ignore
from utils.text_utils import format_email
text_folder_path="./data/raw/gpt"
text_file_name_template="large-762M-k40."
text_file_types=["train","valid","test"]

class GPTDatasetV1(Dataset):
    """Dataset for GPT language modeling from text data"""
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the text into overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class TextDatasetFromCSV(Dataset):
    """Dataset for GPT language modeling from CSV files with text column"""
    def __init__(self, csv_files: List[str], tokenizer, max_length=256, stride=128):
        self.input_ids = []
        self.target_ids = []
        
        # Load and concatenate text from all CSV files
        all_texts = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, engine="python", on_bad_lines='skip', quoting=3)
            if 'text' in df.columns:
                texts = df['text'].dropna().tolist()
                all_texts.extend(texts)
        
        # Join all texts with special token separator
        full_text = "<|endoftext|>".join(all_texts)
        
        # Tokenize
        token_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
        
        # Create sliding window chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

class EmailClassificationDataset(Dataset):
    """Dataset for email classification"""
    def __init__(self, csv_files: List[str], tokenizer, max_length=512):
        self.texts = []
        self.labels = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, engine="python", on_bad_lines='skip', quoting=3)
            
            # Format emails and extract labels
            formatted_emails = df.apply(format_email, axis=1).tolist()
            labels = df["label"].astype(int).tolist()
            
            self.texts.extend(formatted_emails)
            self.labels.extend(labels)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize text
        tokens = self.tokenizer.encode(self.texts[idx], allowed_special={"<|endoftext|>"})
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # Create attention mask
        attention_mask = [1 if token != 0 else 0 for token in tokens]
        
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )



def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

class PhishingEmailDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe.apply(format_email, axis=1).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = torch.tensor(self.labels[idx])
        return input_ids, attention_mask, label


def load_text_only(train_path="large-762M-k40.train.csv",
                   val_path="large-762M-k40.valid.csv",
                   test_path="large-762M-k40.test.csv"):
    """
    Loads only the 'text' column from each CSV and joins all rows into a single string.
    """
    def extract_texts(path):
        df = pd.read_csv(
            path,
            usecols=["text"],
            engine="python",
            on_bad_lines='skip',
            quoting=3
        )
        df.dropna(subset=["text"], inplace=True)
        return df["text"].tolist()

    train_texts = "\n".join(extract_texts(train_path))
    val_texts = "\n".join(extract_texts(val_path))
    test_texts = "\n".join(extract_texts(test_path))

    return train_texts, val_texts, test_texts

def load_csv_files(pattern: str) -> List[str]:
    """Load CSV files matching a pattern"""
    return glob.glob(pattern)

def create_language_model_dataloaders(csv_files: List[str], batch_size=8, max_length=256, 
                                    stride=128, train_ratio=0.8):
    """Create dataloaders for language modeling"""
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Split files into train/val
    train_files = csv_files[:int(len(csv_files) * train_ratio)]
    val_files = csv_files[int(len(csv_files) * train_ratio):]
    
    if not val_files:  # If only one file, use part of it for validation
        val_files = train_files
    
    train_dataset = TextDatasetFromCSV(train_files, tokenizer, max_length, stride)
    val_dataset = TextDatasetFromCSV(val_files, tokenizer, max_length, stride)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, tokenizer

def create_classification_dataloaders(csv_files: List[str], batch_size=8, max_length=512, 
                                    train_ratio=0.8):
    """Create dataloaders for classification"""
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Split files into train/val
    train_files = csv_files[:int(len(csv_files) * train_ratio)]
    val_files = csv_files[int(len(csv_files) * train_ratio):]
    
    if not val_files:
        val_files = train_files
    
    train_dataset = EmailClassificationDataset(train_files, tokenizer, max_length)
    val_dataset = EmailClassificationDataset(val_files, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer