import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def result_summarizer(y_true, y_pred, model_name="Model"):
    """
    Calculates and prints Accuracy, Macro Precision, Macro Recall, and Macro F1 Score,
    and saves a Seaborn Confusion Matrix heatmap to disk as a .png.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n==========================================")
    print(f" Evaluation Results: {model_name}")
    print(f"==========================================")
    print(f" Accuracy        : {acc:.4f} ({acc * 100:.2f}%)")
    print(f" Macro Precision : {prec:.4f}")
    print(f" Macro Recall    : {rec:.4f}")
    print(f" Macro F1 Score  : {f1:.4f}")
    print(f"==========================================\n")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    # Save PNG file
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
    png_filename = f"confusion_matrix_{safe_name}.png"
    plt.savefig(png_filename, dpi=300)
    plt.close()
    print(f" Saved confusion matrix heatmap to '{png_filename}'")
    
    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def main():
    print("1. Loading processed .npy arrays...")
    if not (os.path.exists("X_train.npy") and os.path.exists("X_test.npy") and
            os.path.exists("y_train.npy") and os.path.exists("y_test.npy")):
        raise FileNotFoundError("Processed .npy files not found. Please run '1_data_processing.py' first.")

    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"   X_test shape : {X_test.shape}, y_test shape : {y_test.shape}")

    # Define models
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("K-Neighbors Classifier (k=5)", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ("Decision Tree (max_depth=14)", DecisionTreeClassifier(max_depth=14, random_state=42))
    ]

    results = []

    # Train and evaluate models
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        print(f"Predicting on test set with {name}...")
        y_pred = model.predict(X_test)
        
        metrics = result_summarizer(y_test, y_pred, model_name=name)
        results.append(metrics)

    # Comparison Summary
    print("\n" + "#"*50)
    print(" FINAL MODEL PERFORMANCE COMPARISON")
    print("#"*50)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 56)
    for r in results:
        print(f"{r['model']:<30} | {r['accuracy']*100:>8.2f}% | {r['f1']:>8.4f}")
    print("#"*50 + "\n")

if __name__ == "__main__":
    main()
