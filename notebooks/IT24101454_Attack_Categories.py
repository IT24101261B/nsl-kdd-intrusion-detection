#Feature Engineering (Attack Categories)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Attack Category Creation ---")

#Load data
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class","difficulty"]
df = pd.read_csv("../data/raw/KDDTrain+.txt", names=col_names)

#Implementation
print("\n--- Implementation & Output ---")
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
print("Created 'attack_category' column. Value counts:")
print(df["attack_category"].value_counts())

#EDA: Attack Category Distribution
print("\n--- EDA: Generating Attack Category Distribution Plot ---")
os.makedirs("../results/eda_visualizations", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.countplot(x="attack_category", data=df, order=df["attack_category"].value_counts().index, palette="Set2")
plt.title("Attack Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()

#Save the plot
plot_path = "../results/eda_visualizations/attack_categories.png"
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")
print("\n Script finished.")