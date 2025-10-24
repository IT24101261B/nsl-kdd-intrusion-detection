#Feature Engineering (Binary Label)
import pandas as pd
import matplotlib.pyplot as plt
import os

print("--- Binary Label Creation ---")

#Load data
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class","difficulty"]
df = pd.read_csv("../data/raw/KDDTrain+.txt", names=col_names)

#Implementation
print("\n--- Implementation & Output ---")
df["label_binary"] = (df["class"] != "normal").astype(int)
print("Created 'label_binary' column. Showing 'class' vs. 'label_binary':")
print(df[["class", "label_binary"]].head(10))

#EDA: Binary Class Balance Visualization
print("\n--- EDA: Generating Binary Class Balance Plot ---")
os.makedirs("../results/eda_visualizations", exist_ok=True)

plt.figure(figsize=(6, 4))
df["label_binary"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Normal vs Attack (Binary Classes)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0, 1], ["Normal (0)", "Attack (1)"], rotation=0)
plt.tight_layout()

#Save the plot
plot_path = "../results/eda_visualizations/binary_balance.png"
plt.savefig(plot_path)
plt.close() # Close the plot to prevent it from displaying if not needed

print(f"Plot saved to: {plot_path}")
print("\n Script finished.")