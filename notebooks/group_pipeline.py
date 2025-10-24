#IT2011 Group Assignment: Integrated Preprocessing Pipeline
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import shutil
import kagglehub

print("--- [START] Integrated Group Pipeline ---")

#Automatically downloads the dataset and places it in the correct folder.
print("\n[0/5] Setting up environment and dataset...")

#Define the path for our local raw data
raw_data_dir = "../data/raw"
raw_data_file = "KDDTrain+.txt"
local_file_path = os.path.join(raw_data_dir, raw_data_file)

#Create the data/raw directory if it doesn't exist
os.makedirs(raw_data_dir, exist_ok=True)

#Check if the dataset already exists to avoid re downloading
if not os.path.exists(local_file_path):
    print(f"-> Dataset not found locally. Downloading from Kaggle...")
    try:
        download_path = kagglehub.dataset_download("hassan06/nslkdd")
        print(f"-> Download complete. Files located at: {download_path}")
        source_file_path = os.path.join(download_path, raw_data_file)
        shutil.move(source_file_path, local_file_path)
        print(f"-> Successfully moved '{raw_data_file}' to '{raw_data_dir}'")

    except Exception as e:
        print(f"-> ERROR: Failed to download or move the dataset. Error: {e}")
        print("-> Please check your internet connection and Kaggle API credentials.")
        exit()
else:
    print("-> Dataset already exists locally. Skipping download.")

print("\n[1/5] Loading and Verifying Data...")

col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "class", "difficulty"
]

try:
    df = pd.read_csv(local_file_path, names=col_names)
    print(f"-> Successfully loaded dataset. Shape: {df.shape}")
except FileNotFoundError:
    print(f"-> ERROR: The file was not found at '{local_file_path}'")
    exit()

print(f"-> Found {df.isnull().sum().sum()} missing values and {df.duplicated().sum()} duplicate rows.")

print("\n[2/4] Performing Feature Engineering...")

#Create Binary target
df["label_binary"] = (df["class"] != "normal").astype(int)
print("-> Created binary target column 'label_binary'.")

#Define attack categories and map them
dos_attacks = ["back", "land", "neptune", "pod", "smurf", "teardrop"]
probe_attacks = ["ipsweep", "nmap", "portsweep", "satan"]
r2l_attacks = ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"]
u2r_attacks = ["buffer_overflow", "loadmodule", "perl", "rootkit"]

def map_attack(label):
    if label in dos_attacks: return "DoS"
    elif label in probe_attacks: return "Probe"
    elif label in r2l_attacks: return "R2L"
    elif label in u2r_attacks: return "U2R"
    elif label == "normal": return "Normal"
    else: return "Other"

df["attack_category"] = df["class"].apply(map_attack)
print("-> Created multi-class target column 'attack_category'.")

print("\n[3/4] Applying Preprocessing Techniques...")

#Encode Categorical Variables
cat_cols = ["protocol_type", "service", "flag"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print("-> Applied Label Encoding to categorical features.")

#Scale Numerical Features
#Columns to exclude from scaling are the original labels and the new engineered ones
exclude_from_scaling = {"class", "difficulty", "label_binary", "attack_category"}
num_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c not in exclude_from_scaling]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("-> Applied Standardization to numerical features.")

print("\n[4/4] Saving Final Processed Dataset...")

#Define the output directory and create it if it doesn't exist
output_dir = "../results/outputs"
os.makedirs(output_dir, exist_ok=True)

#Define the final output path
output_path = os.path.join(output_dir, "train_processed.csv")

#Drop the original, high-cardinality 'class' column and 'difficulty' as they are not needed for modeling
df_final = df.drop(columns=["class", "difficulty"])

#Save the final dataframe
df_final.to_csv(output_path, index=False)
print(f"-> Successfully saved processed data to: {output_path}")
print("\n--- [SUCCESS] Pipeline execution comple------------te. ---")