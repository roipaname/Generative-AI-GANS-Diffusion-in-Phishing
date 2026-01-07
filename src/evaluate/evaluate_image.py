

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from src.dataloader.image_loader import load_and_preprocess_data
from config.constant import output_dir


def main():
    # -------------------------------
    # 1. Load Data
    # -------------------------------
    data_dir = "data/phishingimage"
    X, y, class_labels = load_and_preprocess_data(data_dir)

    # One-hot encode labels
    y = to_categorical(y)

    # Train/val/test split (same as training)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Data split -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # -------------------------------
    # 2. Load Trained Model
    # -------------------------------
    model_path = os.path.join(output_dir, "cnn_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # -------------------------------
    # 3. Evaluate Model
    # -------------------------------
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Metrics
    precision = precision_score(y_true_classes, y_pred_classes, average="weighted")
    recall = recall_score(y_true_classes, y_pred_classes, average="weighted")
    f1 = f1_score(y_true_classes, y_pred_classes, average="weighted")

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

    # -------------------------------
    # 4. Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ == "__main__":
    main()
