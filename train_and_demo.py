import os
import warnings

# Suppress TensorFlow logging & warnings for clean presentation terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras
def main():
    print("=" * 65)
    print("  STUDENT PASS / FAIL PREDICTION USING NEURAL NETWORK")
    print("  2-Minute College Demo")
    print("=" * 65)

    # 1. Load Dataset
    csv_file = "students.csv"
    if not os.path.exists(csv_file):
        csv_file = os.path.join(os.path.dirname(__file__), "students.csv")

    df = pd.read_csv(csv_file)
    print(f"\n[Step 1] Loaded Dataset: {csv_file}")
    print(f"Total Student Records: {len(df)}")
    print("\nSample Data (First 5 records):")
    print(df.head().to_string(index=False))

    # 2. Preprocessing
    print("\n" + "-" * 65)
    print("[Step 2] Data Preprocessing")
    
    # Features (X): StudyHours, Attendance, PreviousScore
    # Target (y): Pass = 1, Fail = 0
    X = df[['StudyHours', 'Attendance', 'PreviousScore']].values
    y = df['Result'].map({'Fail': 0, 'Pass': 1}).values

    # Train / Test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling using StandardScaler (mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print("Features normalized with StandardScaler.")

    # 3. Model Architecture
    print("\n" + "-" * 65)
    print("[Step 3] Building the Neural Network Model")
    print("Architecture:")
    print("  - Input Layer:  3 features (StudyHours, Attendance, PreviousScore)")
    print("  - Hidden Layer: 8 neurons, ReLU activation")
    print("  - Output Layer: 1 neuron, Sigmoid activation (Binary Classification)")

    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.Sequential([
        keras.Input(shape=(3,)),
        keras.layers.Dense(8, activation='relu', name="Hidden_Layer"),
        keras.layers.Dense(1, activation='sigmoid', name="Output_Layer")
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    model.summary()

    # 4. Training
    print("\n" + "-" * 65)
    print("[Step 4] Training Neural Network (100 Epochs with Adam optimizer)...")
    model.fit(
        X_train_scaled,
        y_train,
        epochs=20,
        batch_size=5,
        verbose=0
    )
    print("Training Complete!")

    # 5. Evaluation
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print("\n" + "-" * 65)
    print("[Step 5] Evaluation on Test Data")
    print(f"Test Loss (binary_crossentropy): {loss:.4f}")
    print(f"Test Accuracy:                   {accuracy * 100:.2f}%")

    # 6. Demo Predictions on Sample Students
    print("\n" + "-" * 65)
    print("[Step 6] Live Demo Predictions")
    
    test_cases = [
        {"name": "Student A (High Effort)", "features": [7.0, 90.0, 80.0]},
        {"name": "Student B (Low Effort)",  "features": [1.5, 48.0, 32.0]},
        {"name": "Student C (Moderate/Edge)", "features": [4.0, 68.0, 52.0]}
    ]

    for tc in test_cases:
        feat = np.array([tc["features"]])
        feat_scaled = scaler.transform(feat)
        prob = model.predict(feat_scaled, verbose=0)[0][0]
        prediction = "PASS" if prob >= 0.5 else "FAIL"
        
        print(f"\nProfile: {tc['name']}")
        print(f"  Inputs: Study Hours={tc['features'][0]}h | Attendance={tc['features'][1]}% | Prev Score={tc['features'][2]}")
        print(f"  Neural Network Output (Probability): {prob * 100:.2f}%")
        print(f"  Final Decision: >>> {prediction} <<<")

    print("\n" + "=" * 65)
    print("Demo completed successfully!")
    print("=" * 65)

    # Interactive test for evaluator/audience
    try:
        choice = input("\nWould you like to test with custom values? (y/n, default: n): ").strip().lower()
        if choice == 'y':
            while True:
                try:
                    print("\nEnter student details:")
                    hrs = float(input("  Daily Study Hours (e.g., 5.0): "))
                    att = float(input("  Attendance % (e.g., 80.0): "))
                    scr = float(input("  Previous Score (e.g., 65.0): "))

                    feat_scaled = scaler.transform(np.array([[hrs, att, scr]]))
                    prob = model.predict(feat_scaled, verbose=0)[0][0]
                    res = "PASS" if prob >= 0.5 else "FAIL"

                    print(f"  -> Prediction: {res} (Pass Probability: {prob * 100:.2f}%)")
                    
                    again = input("Test another student? (y/n): ").strip().lower()
                    if again != 'y':
                        break
                except ValueError:
                    print("Invalid input. Please enter numbers.")
    except (EOFError, KeyboardInterrupt):
        pass

if __name__ == '__main__':
    main()
