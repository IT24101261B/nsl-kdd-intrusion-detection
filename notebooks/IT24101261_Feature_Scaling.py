#Feature Scaling(Standardization)
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Numerical Feature Scaling ---")

#Load data
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class","difficulty"]
df = pd.read_csv("../data/raw/KDDTrain+.txt", names=col_names)
df_scaled = df.copy()

#Implementation
print("\n--- Implementation & Output ---")
#Identify numeric columns
num_cols = df_scaled.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

print("Original 'src_bytes' stats:")
print(df['src_bytes'].describe())
print("\nScaled 'src_bytes' stats:")
print(df_scaled['src_bytes'].describe())

#EDA: Correlation Heatmap
print("\n--- EDA: Generating Feature Correlation Heatmap ---")
os.makedirs("../results/eda_visualizations", exist_ok=True)

#We need to encode categoricals first to include them in a broad correlation matrix
df_encoded = df.copy()
for col in ["protocol_type", "service", "flag"]:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

#Scale the numeric columns of this encoded dataframe
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
plt.figure(figsize=(12, 10))

#Compute correlation on the scaled and encoded numeric data
corr = df_encoded[num_cols].corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()

#Save the plot
plot_path = "../results/eda_visualizations/corr_heatmap.png"
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")
print("\n Script finished.")