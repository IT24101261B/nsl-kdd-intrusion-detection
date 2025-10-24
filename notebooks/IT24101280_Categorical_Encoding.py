#Handling Categorical Variables
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Categorical Variable Encoding ---")

#Load data
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class","difficulty"]
df = pd.read_csv("../data/raw/KDDTrain+.txt", names=col_names)

#Implementation
print("\n--- Implementation & Output ---")
cat_cols = ["protocol_type", "service", "flag"]
df_encoded = df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    if col == 'protocol_type':
        print(f"Mapping for '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\nOriginal Categorical Data:")
print(df[cat_cols].head())
print("\nEncoded Categorical Data:")
print(df_encoded[cat_cols].head())

#EDA: Top Services by Class
print("\n--- EDA: Generating Top 10 Services Plot ---")
os.makedirs("../results/eda_visualizations", exist_ok=True)
df["label_binary"] = (df["class"] != "normal").astype(int) # Needed for hue

top_services = df["service"].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.countplot(x="service", data=df[df["service"].isin(top_services)],
              order=top_services, hue="label_binary", palette="muted")
plt.title("Top 10 Services - Normal vs Attack")
plt.xlabel("Service")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Label (0=Normal, 1=Attack)")
plt.tight_layout()

#Save the plot
plot_path = "../results/eda_visualizations/top10_services.png"
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")
print("\ Script finished.")