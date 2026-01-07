

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from config.constant import GPT_CONFIG
from src.dataloader.text_loader import load_csv_files,create_classification_dataloaders, create_language_model_dataloaders
from src.utils.metrics import calc_loss_loader
from src.utils.text_utils import text_to_token_ids, token_ids_to_text, generate_text_simple


def evaluate_language_model(model, val_loader, device, tokenizer, start_context="The following text"):
    """Evaluate pre-trained GPT language model"""
    print("\n--- Starting Language Model Evaluation ---")
    model.eval()

    with torch.no_grad():
        print("Calculating validation loss...")
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Validation Loss (Language Model): {val_loss:.4f}")

    # Generate sample
    context_size = model.pos_emb.weight.shape[0]
    print(f"Context size from model: {context_size}")
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    print(f"Encoded start context: {encoded.tolist()}")

    with torch.no_grad():
        print("Generating sample text...")
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("\nGenerated Sample:")
    print(decoded_text.replace("\n", " "))
    print("--- Language Model Evaluation Completed ---\n")


def evaluate_classification_model(model, data_loader, device, class_labels=None, max_batches=None):
    """Evaluate pre-trained GPT classification model with enhanced visualizations"""
    print("\n--- Starting Classification Model Evaluation ---")
    model.eval()
    y_true, y_pred = [],[]

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels, _) in enumerate(data_loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                print(f"\nReached cutoff at batch {max_batches}, stopping evaluation early.")
                break

            print(f"\nProcessing batch {batch_idx}...")
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            print(f"  True labels: {labels.cpu().numpy().tolist()}")
            print(f"  Pred labels: {preds.cpu().numpy().tolist()}")

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    if len(y_true) == 0:
        print("\n⚠️ No batches processed. Exiting without evaluation.\n")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n--- Final Evaluation Metrics ---")
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels if class_labels else None, zero_division=0))

    # --- Confusion Matrix ---
    print("\nGenerating confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # --- Normalized Confusion Matrix ---
    print("Generating normalized confusion matrix plot...")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # --- Per-class metrics bar plot ---
    print("Generating per-class metrics plot...")
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(class_labels)) if class_labels else np.arange(len(precisions))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1-Score')
    plt.xticks(x, class_labels if class_labels else x)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Per-Class Metrics")
    plt.legend()
    plt.show()

    # --- Prediction distribution ---
    print("Generating prediction distribution plot...")
    plt.figure(figsize=(8, 4))
    sns.histplot(y_pred, bins=len(class_labels), kde=False)
    plt.xticks(np.arange(len(class_labels)), class_labels if class_labels else np.arange(len(class_labels)))
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.show()

    print("--- Classification Model Evaluation Completed ---\n")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # Evaluate Language Model
    # -------------------------------
    from src.models.text.gpt_classifier import GPTModel
    
    
    
    print("Loading pre-trained GPT language model...")
    lm_model = GPTModel(GPT_CONFIG).to(device)
    lm_model.load_state_dict(torch.load("outputs/gpt_model.pth", map_location=device))
    lang_csv_files = load_csv_files("./data/language/*.csv")

    train_loader, val_loader, tokenizer = create_language_model_dataloaders(lang_csv_files)
    evaluate_language_model(lm_model, val_loader, device, tokenizer)

    # -------------------------------
    # Evaluate Classification Model
    # -------------------------------
    from src.models.text.gpt_classifier import GPTModel, GPTForClassification
    print("\nLoading pre-trained GPT classification model...")
    gpt_model = GPTModel(GPT_CONFIG)
    cls_model = GPTForClassification(
        gpt_model=gpt_model,
        hidden_size=GPT_CONFIG["emb_dim"],
        num_classes=2
    ).to(device)

    cls_model.load_state_dict(torch.load("outputs/classification_gpt.pth", map_location=device))
    class_csv_files = load_csv_files("./data/phishing/*.csv")

    train_loader, val_loader, tokenizer = create_classification_dataloaders(class_csv_files)
    class_labels = ["Not Phishing", "Phishing"]

    evaluate_classification_model(cls_model, val_loader, device, class_labels=class_labels,max_batches=40)
