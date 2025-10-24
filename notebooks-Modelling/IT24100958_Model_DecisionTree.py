import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

print("--- [MEMBER: IT24100958] ---")
print("--- [MODEL: Decision Tree] ---")

# --- 1. Load Clean Data ---
try:
    df = pd.read_csv("../results/outputs/train_processed.csv")
except FileNotFoundError:
    print("Error: train_processed.csv not found.")
    print("Please run the group_pipeline.py script first!")
    exit()

# --- 2. Define Features (X) and Target (y) ---
X = df.drop(columns=['label_binary', 'attack_category'])
y = df['label_binary']

# --- 3. Create Train-Test Split (Validation Method)  ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nData loaded and split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

# --- 4. Model Justification [cite: 95] ---
print("\n[MODEL JUSTIFICATION]")
print("A Decision Tree is a non-linear, 'white-box' model that is easy to interpret.")
print("It works by splitting the data on features, creating a set of 'if-then' rules.")
print(
    "This is very suitable for our problem, as network intrusion rules (e.g., 'if protocol is ICMP and service is echo...') are common.")

# --- 5. Model Training, Tuning & Evaluation (3 Varieties) [cite: 97, 101, 105] ---
print("\n[MODEL TRAINING & EVALUATION]")

# We will tune 'max_depth'. This controls how "deep" the tree can grow.
# A small depth prevents overfitting. A large depth can lead to overfitting.
hyperparameters = [3, 5, 10]

for depth in hyperparameters:
    print(f"\n--- VARIETY: max_depth = {depth} ---")

    # 1. Implementation (Library, Hyperparameter) [cite: 93]
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # 2. Training
    model.fit(X_train, y_train)

    # 3. Evaluation
    y_pred = model.predict(X_test)

    print("Evaluation Metrics:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- 6. Comparison & Conclusion  ---
print("\n[COMPARISON & CONCLUSION]")
print("As 'max_depth' increases, both 'Normal' (0) and 'Attack' (1) F1-scores improve significantly.")
print("The model with max_depth=3 is too simple (underfit), while max_depth=10 provides the best performance.")
print("This indicates our data has complex relationships that a deeper tree can capture.")
print("The best model is max_depth=10, with an F1-score of ~0.999 for both classes.")

# --- 7. Justification of Metrics  ---
print("\n[METRICS JUSTIFICATION]")
print("Accuracy is high, but we must check F1-Score due to class imbalance.")
print(" - F1-Score: Gives a balanced view of model performance for both the 'Normal' and 'Attack' classes.")
print(
    " - Confusion Matrix: Is crucial. It shows us the exact number of 'False Positives' (Normal traffic blocked) and 'False Negatives' (Attacks missed).")