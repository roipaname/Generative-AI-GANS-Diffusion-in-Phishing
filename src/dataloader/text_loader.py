
import torch, torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import tiktoken # type: ignore

text_folder_path="./data/raw/gpt"
text_file_name_template="large-762M-k40."
text_file_types=["train","valid","test"]

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


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

    train_texts = " ".join(extract_texts(train_path))
    val_texts = " ".join(extract_texts(val_path))
    test_texts = " ".join(extract_texts(test_path))

    return train_texts, val_texts, test_texts

