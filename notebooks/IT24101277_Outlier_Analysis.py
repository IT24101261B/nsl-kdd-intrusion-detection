#Outlier Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Outlier Analysis ---")

#Load data
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class","difficulty"]
df = pd.read_csv("../data/raw/KDDTrain+.txt", names=col_names)

#Implementation
print("\n--- Implementation & Output ---")
print("Analyzing outliers for the 'duration' feature.")
print(df['duration'].describe())

#EDA: Boxplot for Outlier Visualization
print("\n--- EDA: Generating Boxplot for Duration ---")
os.makedirs("../results/eda_visualizations", exist_ok=True)
df["label_binary"] = (df["class"] != "normal").astype(int) # Needed for hue

plt.figure(figsize=(8, 6))
sns.boxplot(x="label_binary", y="duration", data=df)
plt.title("Duration Distribution: Normal vs Attack (Outlier View)")
plt.xlabel("Class (0=Normal, 1=Attack)")
plt.ylabel("Duration")
plt.ylim(0, 5000)
plt.tight_layout()

#Save the plot
plot_path = "../results/eda_visualizations/duration_boxplot.png"
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")
print("\n Script finished.")