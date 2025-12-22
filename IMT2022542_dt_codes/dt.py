import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import xml.etree.ElementTree as ET
import os

def get_config_params(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    iat_val = None
    qos_val = None
    for app in root.iter('APPLICATION'):
        if app.attrib.get('NAME') == 'App1_INTERACTIVE_GAMING':
            qos_val = app.attrib.get('QOS')
            for child in app:
                if child.tag == 'DL_INTER_ARRIVAL_TIME':
                    iat_val = child.attrib.get('VALUE')
                    break
            break
    return iat_val, qos_val

def run_prediction_model(config_path = r"C:\Users\admin\Documents\NetSim\Workspaces\workspace\IMT2022542_dt\configuration.netsim",
        save_dir = r"C:\Users\adim\Desktop\IMT2022542_dt_codes\IMT2022542_dt_results"
):

    dt = pd.read_csv("dt_dataset.csv")
    dt.head()
    dt.info()

    dt = dt.dropna()

    null_count = dt.isnull().sum()
    print(null_count[null_count>0])

    dt.shape

    num_zeros = (dt['Throughput']==0).sum()

    total_samples = len(dt)

    print(f"Number of zero throughput samples: {num_zeros}")
    print(f"Percentage of zeros: {num_zeros / total_samples * 100:.2f}%")
    dt['Throughput'].mean()
    dt['Latency'].mean()

    columns_to_check = ["LTE SINR", "NR CQI", "NR MCS", "Throughput","Latency"]
    dt_iqr = dt.copy()

    for i in columns_to_check:
        Q1 = dt_iqr[i].quantile(0.25)
        Q3 = dt_iqr[i].quantile(0.75)
        IQR = Q3-Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

    dt_iqr = dt_iqr[(dt_iqr[i] >= lower) & (dt_iqr[i] <= upper)]

    print(f"Original size: {dt.shape}")
    print(f"Filtered size: {dt_iqr.shape}")

    features = ["LTE SINR", "NR MCS"]
    target = ["Throughput"]

    X = dt_iqr[features]
    y = dt_iqr[target]

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)

    X_val_scaled = scaler_x.transform(X_val_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)

    sequence_length = 40

    def create_sequences(X, y, seq_len=40):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)

    current_iat, current_qos = get_config_params(config_path)

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, sequence_length)

    X_train.shape

    cnn_lstm_model = Sequential()
    cnn_lstm_model.add(Conv1D(128, 3, activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2])))
    cnn_lstm_model.add(Conv1D(64, 3, activation='relu', padding='same'))
    cnn_lstm_model.add(MaxPooling1D(2))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(LSTM(64, return_sequences=True))
    cnn_lstm_model.add(LSTM(32))
    cnn_lstm_model.add(Dense(1))

    cnn_lstm_model.compile(optimizer='adam', loss='mse')
    cnn_lstm_model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    cnn_lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

    # Predict and evaluate
    preds = cnn_lstm_model.predict(X_val)
    preds_inverse = scaler_y.inverse_transform(preds)
    y_val_inverse = scaler_y.inverse_transform(y_val)

    # Compute metrics
    mae = mean_absolute_error(y_val_inverse, preds_inverse)
    mse = mean_squared_error(y_val_inverse, preds_inverse)
    r2 = r2_score(y_val_inverse, preds_inverse)

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.8f}")
    print(f"R²:   {r2:.4f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)

    # Append metrics to central CSV
    metrics_file = os.path.join(save_dir, "dt_metrics.csv")
    row = pd.DataFrame([{
        "IAT": current_iat,
        "QoS": current_qos,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }])
    if os.path.exists(metrics_file):
        row.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        row.to_csv(metrics_file, index=False)

    # Save predictions
    preds_file = os.path.join(save_dir, f"dt_predictions_iat{current_iat}_qos{current_qos}.csv")
    df_results = pd.DataFrame({
        "predicted_throughput": preds_inverse.flatten(),
        "actual_throughput": y_val_inverse.flatten()
    })
    df_results.to_csv(preds_file, index=False)

    # Save plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_val_inverse, label="Actual Throughput", color="blue")
    plt.plot(preds_inverse, label="Predicted Throughput", color="red")
    plt.xlabel("Sample")
    plt.ylabel("Throughput")
    plt.title(f"Throughput Prediction (IAT={current_iat}, QoS={current_qos})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"throughput_plot_iat{current_iat}_qos{current_qos}.png"))
    plt.close()

    return y_val_inverse.flatten(), preds_inverse.flatten(), mse, r2