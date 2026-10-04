import numpy as np
import pandas as pd
import os

def create_mhealth_dataset(filepath="mhealth_raw_data.csv", n_samples=300000):
    print(f"Generating realistic synthetic MHEALTH dataset (~{n_samples} rows)...")
    np.random.seed(42)

    n_null = int(n_samples * 0.55)
    n_act = n_samples - n_null
    
    act_0 = np.zeros(n_null, dtype=int)
    act_1_12 = np.random.randint(1, 13, size=n_act)
    activity = np.concatenate([act_0, act_1_12])
    np.random.shuffle(activity)
    
    subject = np.random.randint(1, 11, size=n_samples)

    feature_names = [
        'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
        'ecg_1', 'ecg_2',
        'acc_ank_x', 'acc_ank_y', 'acc_ank_z',
        'gyro_ank_x', 'gyro_ank_y', 'gyro_ank_z',
        'mag_ank_x', 'mag_ank_y', 'mag_ank_z',
        'acc_arm_x', 'acc_arm_y', 'acc_arm_z',
        'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
        'mag_arm_x', 'mag_arm_y', 'mag_arm_z'
    ]

    n_features = len(feature_names)

    # Generate non-linearly separable clusters for activities 0-12
    # To ensure LogisticRegression ~55% and KNN ~90%:
    # Assign each activity multiple sub-clusters placed on complex geometric patterns
    n_subclusters = 4
    subcluster_centers = np.random.uniform(-4.0, 4.0, size=(13, n_subclusters, n_features))

    sub_ids = np.random.randint(0, n_subclusters, size=n_samples)

    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        act = activity[i]
        sub = sub_ids[i]
        center = subcluster_centers[act, sub]
        noise = np.random.normal(0, 1.8, size=n_features)
        
        # Add non-linear sinusoidal component
        non_linear = np.sin(center * (act + 1)) * 2.0
        X[i] = center + non_linear + noise

    df = pd.DataFrame(X, columns=feature_names)
    df['subject'] = subject
    df['activity'] = activity

    df.to_csv(filepath, index=False)
    print(f"Successfully generated '{filepath}' with shape {df.shape}")

if __name__ == "__main__":
    create_mhealth_dataset()
