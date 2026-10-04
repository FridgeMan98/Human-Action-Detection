import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False

def export_model():
    print("1. Loading training and testing arrays...")
    if not (os.path.exists("X_train.npy") and os.path.exists("y_train.npy")):
        raise FileNotFoundError("Training data arrays (X_train.npy, y_train.npy) not found. Run 1_data_processing.py first.")

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    print("2. Training winning model: KNeighborsClassifier(n_neighbors=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)

    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    print(f"   Train Accuracy: {train_acc * 100:.2f}%")
    print(f"   Test Accuracy : {test_acc * 100:.2f}%")

    model_filename = "best_knn_model.pkl"
    print(f"3. Exporting trained model to '{model_filename}'...")
    if USE_JOBLIB:
        joblib.dump(knn, model_filename)
        print("   Model exported successfully using joblib.")
    else:
        with open(model_filename, "wb") as f:
            pickle.dump(knn, f)
        print("   Model exported successfully using standard pickle.")

def create_requirements_file():
    print("4. Generating requirements.txt...")
    req_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
seaborn>=0.12.0
matplotlib>=3.6.0
joblib>=1.2.0
"""
    with open("requirements.txt", "w") as f:
        f.write(req_content)
    print("   'requirements.txt' created successfully.")

def main():
    export_model()
    create_requirements_file()
    print("\nModel export & project wrap-up completed successfully!")

if __name__ == "__main__":
    main()
