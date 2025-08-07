from utils.metrics import calc_loss_batch,calc_loss_loader
from utils.text_utils import text_to_token_ids,token_ids_to_text,generate_text_simple
import torch


def train_gpt_language_model(model, train_loader, val_loader, optimizer, device, 
                           num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """Train GPT model for language modeling"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Generate sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def train_classification_model(model, train_loader, val_loader, optimizer, device, num_epochs):
    """Train model for classification"""
    train_losses, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Evaluate on validation set
        val_accuracy = evaluate_classification_model(model, val_loader, device)
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

    return train_losses, val_accuracies

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate language model"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def evaluate_classification_model(model, data_loader, device):
    """Evaluate classification model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    model.train()
    return correct / total

def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print sample text"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_language_model_from_csvs(csv_pattern: str, model_config: dict, device: str = "cuda"):
    """Complete pipeline for training language model from CSV files"""
    
    # Load CSV files
    csv_files = load_csv_files(csv_pattern)
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    # Create dataloaders
    train_loader, val_loader, tokenizer = create_language_model_dataloaders(csv_files)
    
    # Initialize model
    from paste import GPTModel  # Import your GPT model
    model = GPTModel(model_config).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Train model
    train_losses, val_losses, tokens_seen = train_gpt_language_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        eval_freq=100,
        eval_iter=25,
        start_context="The following text",
        tokenizer=tokenizer
    )
    
    return model, train_losses, val_losses, tokens_seen

def train_classification_model_from_csvs(csv_pattern: str, gpt_config: dict, device: str = "cuda"):
    """Complete pipeline for training classification model from CSV files"""
    
    # Load CSV files
    csv_files = load_csv_files(csv_pattern)
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    # Create dataloaders
    train_loader, val_loader, tokenizer = create_classification_dataloaders(csv_files)
    
    # Initialize base GPT model
    from paste import GPTModel, GPTForClassification  # Import your models
    gpt_model = GPTModel(gpt_config)
    
    # Create classification model
    model = GPTForClassification(
        gpt_model=gpt_model,
        hidden_size=gpt_config['emb_dim'],
        num_classes=2
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Train model
    train_losses, val_accuracies = train_classification_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=3
    )
    
    return model, train_losses, val_accuracies

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Example 1: Train language model
    print("Training language model...")
    # model, train_losses, val_losses, tokens_seen = train_language_model_from_csvs(
    #     csv_pattern="./data/text_data/*.csv",
    #     model_config=GPT_CONFIG,
    #     device=device
    # )
    
    # Example 2: Train classification model
    print("Training classification model...")
    # model, train_losses, val_accuracies = train_classification_model_from_csvs(
    #     csv_pattern="./data/email_data/*.csv",
    #     model_config=GPT_CONFIG,
    #     device=device
    # )