import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def main():
    csv_file = "mhealth_raw_data.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset file '{csv_file}' not found.")

    print(f"1. Loading dataset: '{csv_file}'...")
    df = pd.read_csv(csv_file)
    print(f"   Initial dataset shape: {df.shape}")

    print("2. Handling class imbalance...")
    df_null = df[df['activity'] == 0]
    df_non_null = df[df['activity'] != 0]

    n_sample = min(40000, len(df_null))
    print(f"   Activity 0 row count before sampling: {len(df_null)}")
    print(f"   Sampling {n_sample} rows from activity == 0...")
    df_null_sampled = df_null.sample(n=n_sample, random_state=42)

    df_balanced = pd.concat([df_null_sampled, df_non_null], axis=0).reset_index(drop=True)
    print(f"   Balanced dataset shape: {df_balanced.shape}")
    print(f"   Activity distribution:\n{df_balanced['activity'].value_counts().sort_index()}")

    print("3. Defining features X and target y...")
    X = df_balanced.drop(columns=['activity', 'subject'])
    y = df_balanced['activity']

    print("4. Performing train-test split (test_size=0.25, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"   X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    print("5. Scaling features using RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("6. Saving array artifacts (.npy)...")
    np.save("X_train.npy", X_train_scaled)
    np.save("X_test.npy", X_test_scaled)
    np.save("y_train.npy", y_train.to_numpy())
    np.save("y_test.npy", y_test.to_numpy())

    print("Data processing pipeline finished successfully. Artifacts saved:")
    print("   - X_train.npy")
    print("   - X_test.npy")
    print("   - y_train.npy")
    print("   - y_test.npy")

if __name__ == "__main__":
    main()
