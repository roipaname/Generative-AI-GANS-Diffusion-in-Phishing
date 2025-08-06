import torch

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=40):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # (1, context_size)

        with torch.no_grad():
            logits = model(idx_cond)  # (1, context, vocab_size)

        logits = logits[:, -1, :]  # (1, vocab_size)
        logits = logits / temperature  # apply temperature

        # Top-k filtering
        if top_k is not None:
            top_logits, top_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_logits, dim=-1)
            next_token = top_indices.gather(1, torch.multinomial(probs, num_samples=1))
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())