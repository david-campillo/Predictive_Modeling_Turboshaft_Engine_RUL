"""
Predictive Maintenance utilizing LSTM for Remaining Useful Life (RUL) estimation on CMAPSS data (FD001).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

import check_residual_normality

# Constants
SEQUENCE_LENGTH = 30
COLUMNS = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

def load_data():
    """Load train, test, and RUL datasets."""
    train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=COLUMNS)
    test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=COLUMNS)
    rul_test = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    
    # Calculate target RUL for training data
    train['RUL'] = train.groupby('unit_id')['time_cycles'].transform(lambda x: x.max() - x)
    return train, test, rul_test

def preprocess_data(train, test):
    """Normalize features and drop sensors with near-zero variance."""
    sensor_cols = [c for c in COLUMNS if 'sensor' in c]
    variances = train[sensor_cols].var()
    drop_sensors = variances[variances < 0.01].index.tolist()
    print(f"Dropping {len(drop_sensors)} constant sensors: {drop_sensors}")

    feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                   [s for s in sensor_cols if s not in drop_sensors]
    print(f"Using {len(feature_cols)} features...")

    # Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    
    train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])
    
    return train_scaled, test_scaled, feature_cols

def create_sequences(df, feature_cols, seq_len):
    """Create windowed sequences for LSTM training."""
    sequences, targets = [], []

    for unit_id in df['unit_id'].unique():
        engine_data = df[df['unit_id'] == unit_id]
        features = engine_data[feature_cols].values
        rul_values = engine_data['RUL'].values

        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i : i + seq_len])
            targets.append(rul_values[i + seq_len - 1])

    return np.array(sequences), np.array(targets)

def prepare_test_data(test, rul_test, feature_cols, seq_len):
    """Extract the last sequence for each engine in the test set to predict final RUL."""
    X_test = []
    y_test = rul_test['RUL'].values

    for unit_id in test['unit_id'].unique():
        engine_data = test[test['unit_id'] == unit_id]
        features = engine_data[feature_cols].values

        # Pad with zeros if the engine's data length is less than the sequence length
        if len(features) < seq_len:
            pad = np.zeros((seq_len - len(features), len(feature_cols)))
            features = np.vstack([pad, features])

        X_test.append(features[-seq_len:])

    return np.array(X_test), np.array(y_test)

def build_model(input_shape):
    """Build and compile the LSTM architecture."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_results(history, y_test, y_pred_test, rmse_test):
    """Visualize training history and prediction scatter plot."""
    plt.figure(figsize=(12, 4))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Actual vs Predicted Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    
    # Perfect prediction line
    max_val = max(y_test.max(), y_pred_test.max())
    if np.isnan(max_val):
        max_val = 150 # fallback
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(f'Test Predictions (RMSE={rmse_test:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results_lstm.png', dpi=100)
    print("Saved training results plot to 'training_results_lstm.png'")

def main():
    print("Loading data...")
    train, test, rul_test = load_data()

    print("Preprocessing data...")
    train_scaled, test_scaled, feature_cols = preprocess_data(train, test)

    print("Generating sequences...")
    X_train, y_train = create_sequences(train_scaled, feature_cols, SEQUENCE_LENGTH)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    X_test, y_test = prepare_test_data(test_scaled, rul_test, feature_cols, SEQUENCE_LENGTH)
    print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.summary()

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=128,
        verbose=1
    )

    print("Evaluating model...")
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\nFinal Results:")
    print(f"Train RMSE: {rmse_train:.2f} cycles")
    print(f"Test RMSE:  {rmse_test:.2f} cycles")

    check_residual_normality.check_residual_normality(y_test, y_pred_test)

    print("Plotting results...")
    plot_results(history, y_test, y_pred_test, rmse_test)

if __name__ == '__main__':
    main()
