"""
Predictive Maintenance utilizing XGBoost for Remaining Useful Life (RUL) estimation on CMAPSS data (FD001).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import check_residual_normality

# Constants
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

    # For XGBoost, using 'time_cycles' alongside other variables generally improves predictions
    feature_cols = ['time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                   [s for s in sensor_cols if s not in drop_sensors]
    print(f"Using {len(feature_cols)} features...")

    # Although XGBoost doesn't strictly require normalization, keeping it consistent with LSTM preprocessing
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    
    train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])
    
    return train_scaled, test_scaled, feature_cols

def prepare_training_data(df, feature_cols):
    """Extract features and targets for training."""
    X_train = df[feature_cols].values
    y_train = df['RUL'].values
    return X_train, y_train

def prepare_test_data(test, rul_test, feature_cols):
    """Extract the very last cycle for each engine in the test set to predict final RUL."""
    # Test set targets correspond back to the final state/last cycle of each unit
    X_test = test.groupby('unit_id').last()[feature_cols].values
    y_test = rul_test['RUL'].values
    return X_test, y_test

def build_model():
    """Instantiate the XGBoost regressor."""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    return model

def plot_results(y_test, y_pred_test, rmse_test):
    """Visualize actual vs predicted scatter plot."""
    plt.figure(figsize=(6, 5))

    plt.scatter(y_test, y_pred_test, alpha=0.5)
    
    # Perfect prediction line
    #max_val = max(y_test.max(), y_pred_test.max())
    max_val = y_test.max()
    if np.isnan(max_val):
        max_val = 150 # fallback
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(f'XGBoost Test Predictions (RMSE={rmse_test:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results_xgboost.png', dpi=100)
    print("Saved training results plot to 'training_results_xgboost.png'")

def main():
    print("Loading data...")
    train, test, rul_test = load_data()

    print("Preprocessing data...")
    train_scaled, test_scaled, feature_cols = preprocess_data(train, test)

    print("Preparing train and test sets...")
    X_train, y_train = prepare_training_data(train_scaled, feature_cols)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    X_test, y_test = prepare_test_data(test_scaled, rul_test, feature_cols)
    print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

    print("Building XGBoost model...")
    model = build_model()

    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=10
    )

    print("Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\nFinal Results:")
    print(f"Train RMSE: {rmse_train:.2f} cycles")
    print(f"Test RMSE:  {rmse_test:.2f} cycles")

    check_residual_normality.check_residual_normality(y_test, y_pred_test)

    print("Plotting results...")
    plot_results(y_test, y_pred_test, rmse_test)

if __name__ == '__main__':
    main()

