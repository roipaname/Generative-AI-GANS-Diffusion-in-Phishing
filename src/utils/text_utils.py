import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Simple text generation function"""
    # Ensure idx has batch dimension
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)  # shape: (1, seq_len)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # slice last context_size tokens on dim=1
        with torch.no_grad():
            logits = model(idx_cond)  # input shape: (batch=1, seq_len=context_size)
        logits = logits[:, -1, :]  # logits for last token: shape (1, vocab_size)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # shape (1, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # concat on sequence length dim=1

    return idx

def format_email(row):
    """Format email data into a single text string - handles both formats"""
    # Check if we have the full email format
    if 'sender' in row and 'receiver' in row:
        return f"From: {row['sender']}\nTo: {row['receiver']}\nDate: {row['date']}\nSubject: {row['subject']}\nBody: {row['body']}\nURLs: {row.get('urls', '')}"
    # Handle simple format with just subject, body, label
    else:
        return f"Subject: {row['subject']}\nBody: {row['body']}"

def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """Convert text to token IDs"""
    return torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))

def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """Convert token IDs back to text"""
    # Flatten the tensor before converting to a list
    return tokenizer.decode(token_ids.flatten().tolist())
