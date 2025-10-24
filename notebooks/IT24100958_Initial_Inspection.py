#Data Loading & Initial Inspection
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Initial Data Inspection ---")

#Define column names for the NSL-KDD dataset
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class","difficulty"]

#Load the dataset from the relative path
try:
    df = pd.read_csv("../data/raw/KDDTrain+.txt", names=col_names)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: KDDTrain+.txt not found. Make sure it is in the 'data/raw/' folder.")
    exit()

#Implementation
print("\n--- Implementation & Output ---")
print("First 5 rows of the dataset:")
print(df.head())

print(f"\nDataset Shape: {df.shape}")
print(f"Total Missing Values: {df.isnull().sum().sum()}")
print(f"Total Duplicate Rows: {df.duplicated().sum()}")

#EDA Visualization: Class Distribution Overview
print("\n--- EDA: Generating and saving class distribution plot... ---")

#Ensure the output directory exists
os.makedirs("../results/eda_visualizations", exist_ok=True)

#Get the top 10 class frequencies
class_counts = df['class'].value_counts().head(10)

#Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title('Top 10 Connection Class Frequencies')
plt.xlabel('Class Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

#Save the figure
plot_path = "../results/eda_visualizations/class_distribution.png"
plt.savefig(plot_path)
plt.close()

print(f"-> Plot saved to: {plot_path}")
print("\n Script finished.")