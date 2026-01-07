
import glob
import torch, torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import tiktoken # type: ignore
import pandas as pd # type: ignore
from src.utils.text_utils import format_email
from typing import List
import os
import numpy as np
text_folder_path="./data/raw/gpt"
text_file_name_template="large-762M-k40."
text_file_types=["train","valid","test"]

class GPTDatasetV1(Dataset):
    """
    Dataset for GPT-style language modeling using plain text data.

    This class tokenizes raw text and generates input-target pairs
    using a sliding window approach. Each input sequence is a chunk of 
    `max_length` tokens, and each target sequence is the same chunk shifted 
    by one position (next-token prediction).

    Args:
        txt (str): The raw text data.
        tokenizer: A tokenizer with an `encode` method (e.g., tiktoken, HuggingFace).
        max_length (int): Maximum sequence length for model input.
        stride (int): Step size for the sliding window over tokenized text.

    Attributes:
        input_ids (List[torch.Tensor]): List of tokenized input sequences.
        target_ids (List[torch.Tensor]): List of tokenized target sequences (shifted).
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text into token IDs
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Create sliding windows of size `max_length`
        # Each window shifts by `stride` tokens
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]                  # Input sequence
            target_chunk = token_ids[i + 1: i + max_length + 1]        # Target = shifted by 1
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Return number of training samples available."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Retrieve input-target pair at index `idx`."""
        return self.input_ids[idx], self.target_ids[idx]


class TextDatasetFromCSV(Dataset):
    """
    Optimized Dataset for GPT-style language modeling from CSV files.

    This dataset loads text data from one or more CSV files (with a "text" column),
    concatenates them, tokenizes, and constructs input-target pairs for language
    modeling using vectorized sliding windows. Optionally, it caches the preprocessed
    dataset to speed up future runs.

    Args:
        csv_files (List[str]): List of CSV file paths containing text data.
        tokenizer: A tokenizer with an `encode` method.
        max_length (int, optional): Maximum sequence length. Default = 256.
        stride (int, optional): Step size for sliding window. Default = 128.
        cache_file (str, optional): Path to save/load cached dataset.

    Attributes:
        input_ids (torch.Tensor): Tensor of input sequences.
        target_ids (torch.Tensor): Tensor of target sequences.
    """

    def __init__(self, csv_files: List[str], tokenizer, max_length=256, stride=128, cache_file=None):
        self.input_ids, self.target_ids = [], []

        # If a cached version exists, load it instead of rebuilding
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            self.input_ids, self.target_ids = torch.load(cache_file)
            return

        # Collect text from all CSVs
        all_texts = []
        for csv_file in csv_files:
            # Load only the "text" column; skip bad lines
            df = pd.read_csv(csv_file, engine="python", on_bad_lines="skip", quoting=3, usecols=["text"])
            texts = df["text"].dropna().tolist()
            all_texts.extend(texts)

        # Concatenate all texts with a special end-of-text token
        full_text = "<|endoftext|>".join(all_texts)

        # Tokenize the entire dataset
        token_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})

        # Use NumPy's stride tricks for efficient overlapping windows
        arr = np.array(token_ids, dtype=np.int64)
        shape = ((arr.size - max_length) // stride + 1, max_length)   # Number of windows × window size
        strides = (arr.strides[0] * stride, arr.strides[0])           # Define step sizes
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

        # Convert to PyTorch tensors
        self.input_ids = torch.tensor(windows[:-1])   # Inputs
        self.target_ids = torch.tensor(windows[1:])   # Targets (shifted by 1)

        # Save preprocessed dataset to cache for faster reloads
        if cache_file:
            torch.save((self.input_ids, self.target_ids), cache_file)

    def __len__(self):
        """Return number of training samples available."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Retrieve input-target pair at index `idx`."""
        return self.input_ids[idx], self.target_ids[idx]

#HELPER FUNCTION
def safe_int(val):
    try:
        return int(val)
    except:
        return None
    
#helper function to load CSV files
def load_and_filter_csv(csv_file):
    df = pd.read_csv(csv_file, engine="python", on_bad_lines='skip', quoting=3)

    # Check for required columns
    if not {'subject', 'body', 'label'}.issubset(df.columns):
        print(f"Warning: {csv_file} missing required columns")
        return None, None

    # Drop rows with missing label
    df = df.dropna(subset=['label'])

    # Convert label to int safely
    df['label'] = df['label'].apply(safe_int)

    # Drop rows where label conversion failed (None)
    df = df.dropna(subset=['label'])

    # Keep only rows where label is 0 or 1
    df = df[df['label'].isin([0, 1])]

    # Optionally, drop rows with missing subject or body
    df = df.dropna(subset=['subject', 'body'])

    # Format emails for text input (assumes you have a function format_email)
    texts = df.apply(format_email, axis=1).tolist()

    labels = df['label'].astype(int).tolist()

    print(f"Loaded {len(texts)} valid samples from {csv_file} (only labels 0/1 kept)")

    return texts, labels


# Dataset for supervised email classification
class EmailClassificationDataset(Dataset):
    """
    Dataset for email classification tasks (binary or multi-class).

    This class loads email texts and their corresponding labels from multiple CSV files.
    It supports tokenization, sequence padding/truncation, and returns samples formatted 
    for PyTorch models.

    Assumes that a helper function `load_and_filter_csv(csv_file)` is defined elsewhere 
    to extract text and label lists from a CSV file. The function should return:
        - texts (List[str]): list of email texts
        - labels (List[int]): corresponding labels (e.g., 0/1 for binary classification)

    Args:
        csv_files (List[str]): List of CSV file paths containing email data.
        tokenizer: Tokenizer with an `encode` method (e.g., GPT tokenizer, HuggingFace).
        max_length (int, optional): Maximum sequence length for tokenized text. Default = 512.

    Attributes:
        texts (List[str]): Loaded email texts.
        labels (List[int]): Corresponding classification labels.
        tokenizer: The tokenizer used for encoding text.
        max_length (int): Maximum length for tokenized sequences (with padding/truncation).
    """

    def __init__(self, csv_files: List[str], tokenizer, max_length=512):
        self.texts = []
        self.labels = []

        # Load and filter data from multiple CSVs
        for csv_file in csv_files:
            try:
                texts, labels = load_and_filter_csv(csv_file)
                if texts is None or labels is None:
                    continue
                self.texts.extend(texts)
                self.labels.extend(labels)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        # If no valid data is found, raise error
        if not self.texts:
            raise ValueError("No valid data found in CSV files")

        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.texts)} email samples from {len(csv_files)} files "
              f"(only labels 0/1 kept)")

    def __len__(self):
        """Return total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve one email sample at index `idx`.

        Returns:
            tokens (torch.LongTensor): Padded/truncated token IDs of the email text.
            attention_mask (torch.LongTensor): Mask with 1s for non-padding tokens, 0s for padding.
            label (torch.LongTensor): Classification label for the email.
            raw_text (str): Original email text.
        """
        # Tokenize email text into token IDs
        tokens = self.tokenizer.encode(
            self.texts[idx],
            allowed_special={"<|endoftext|>"}
        )

        # Truncate if too long, pad if too short
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        # Build attention mask (1 = token present, 0 = padding)
        attention_mask = [1 if token != 0 else 0 for token in tokens]

        return (
            torch.tensor(tokens, dtype=torch.long),           # Token IDs
            torch.tensor(attention_mask, dtype=torch.long),   # Attention mask
            torch.tensor(self.labels[idx], dtype=torch.long), # Label
            self.texts[idx]                                   # Original text (for debugging/inspection)
        )

#test dataloader
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
    """
    Dataset for phishing email classification.

    This dataset prepares email texts and labels for use with a transformer-based model
    (e.g., BERT, GPT, DistilBERT). It handles tokenization, padding, and truncation to a
    fixed sequence length.

    Args:
        dataframe (pd.DataFrame): Pandas DataFrame containing email data.
                                  Must include a 'label' column and fields required by `format_email`.
        tokenizer: HuggingFace-style tokenizer with `__call__` method.
        max_length (int, optional): Maximum sequence length for tokenized text. Default = 512.

    Attributes:
        texts (List[str]): List of formatted email texts.
        labels (List[int]): Corresponding classification labels (0 = benign, 1 = phishing).
        tokenizer: Tokenizer used for encoding.
        max_length (int): Sequence length for padding/truncation.
    """

    def __init__(self, dataframe, tokenizer, max_length=512):
        # Apply formatting function to each row (e.g., subject + body concatenation)
        self.texts = dataframe.apply(format_email, axis=1).tolist()
        
        # Extract labels and ensure they are integers
        self.labels = dataframe["label"].astype(int).tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve one email sample (tokenized).

        Returns:
            input_ids (torch.LongTensor): Token IDs (padded/truncated to max_length).
            attention_mask (torch.LongTensor): Attention mask (1 = token, 0 = padding).
            label (torch.LongTensor): Corresponding label (0/1).
        """
        # Tokenize text with padding and truncation
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",       # Pad sequences to max_length
            truncation=True,            # Truncate longer sequences
            max_length=self.max_length, # Fixed sequence length
            return_tensors="pt"         # Return as PyTorch tensors
        )

        # Squeeze removes the extra batch dimension (shape: [1, max_length] → [max_length])
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Convert label to tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)

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
#retuirns pattern files
def load_csv_files(pattern: str) -> List[str]:
    """Load CSV files matching a pattern"""
    return glob.glob(pattern)
"""Create dataloaders for language modeling"""
def create_language_model_dataloaders(csv_files: List[str], batch_size=8, max_length=256, 
                                    stride=128, train_ratio=0.8):
    """Create dataloaders for language modeling"""
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"found files {csv_files}")
    
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
"""Create dataloaders for classification"""
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