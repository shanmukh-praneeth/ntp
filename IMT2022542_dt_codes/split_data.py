import pandas as pd

# File paths
radio_file = r"C:\Users\admin\Documents\NetSim\Workspaces\workspace\IMT2022542_dt\log\LTENR_Radio_Measurements_Log.csv"
app_file = r"C:\Users\admin\Documents\NetSim\Workspaces\workspace\IMT2022542_dt\log\Application_Packet_Log.csv"
output_file = "dt_dataset.csv"

# Read both CSV files
radio_df = pd.read_csv(radio_file)
app_df = pd.read_csv(app_file)

# Extract required columns and rename
radio_features = radio_df[["SINR(dB)", "CQI Index", "MCS Index"]].rename(columns={
    "SINR(dB)": "LTE SINR",
    "CQI Index": "NR CQI",
    "MCS Index": "NR MCS"
})

app_features = app_df[["Throughput(Mbps)"]].rename(columns={
    "Throughput(Mbps)": "Throughput"
})

# Make number of rows equal by trimming to min length
min_len = min(len(radio_features), len(app_features))
radio_features = radio_features.iloc[:min_len].reset_index(drop=True)
app_features = app_features.iloc[:min_len].reset_index(drop=True)

# Combine into one DataFrame
dt_dataset = pd.concat([radio_features, app_features], axis=1)

# Save to CSV
dt_dataset.to_csv(output_file, index=False)

print(f"Dataset saved as {output_file} with {len(dt_dataset)} rows.")