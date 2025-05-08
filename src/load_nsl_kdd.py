import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Function to preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path, header=None)
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]
    df.columns = columns

    # Convert labels: 0 for normal, 1 for attack
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Encode categorical columns manually
    def encode_column(col):
        unique_vals = sorted(df[col].unique())
        val_map = {val: idx for idx, val in enumerate(unique_vals)}
        return df[col].map(val_map)

    for col in ['protocol_type', 'service', 'flag']:
        df[col] = encode_column(col)

    # Drop 'difficulty' column
    df.drop(columns=['difficulty'], inplace=True)

    # Separate features and labels
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    # Normalize features manually (min-max normalization)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)

    return X, y

# Load and preprocess training data
print("Loading training dataset...")
X_train, y_train = preprocess_data("../data/KDDTrain+.txt")

# Manual train-test split (80/20)
split_index = int(0.8 * len(X_train))
X_train, X_test = X_train[:split_index], X_train[split_index:]
y_train, y_test = y_train[:split_index], y_train[split_index:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Compute class imbalance ratio
num_pos = np.sum(y_train == 1)
num_neg = np.sum(y_train == 0)
pos_weight = torch.tensor([num_neg / (num_pos + 1e-8)], dtype=torch.float32)

# Define a simple neural network
class IDSModel(nn.Module):
    def __init__(self, input_dim):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Initialize model, loss (with pos_weight), and optimizer
model = IDSModel(X_train.shape[1])
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Split data for validation
val_split_index = int(0.8 * len(X_train_tensor))
X_train_val, X_val = X_train_tensor[:val_split_index], X_train_tensor[val_split_index:]
y_train_val, y_val = y_train_tensor[:val_split_index], y_train_tensor[val_split_index:]

# Training loop with early stopping
print("Starting training...\n")
epochs = 100  # More epochs
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_val)
    loss = criterion(outputs, y_train_val)
    if torch.isnan(loss):
        raise ValueError("NaN loss detected!")
    loss.backward()
    optimizer.step()

    # Validation loss for early stopping
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_IDSModel.pth")  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs without improvement.")
            early_stop = True
            break

# Load the best model after early stopping
if early_stop:
    model.load_state_dict(torch.load("best_IDSModel.pth"))

# Save the model after training
torch.save(model.state_dict(), "IDSModel.pth")
print("Model saved to 'IDSModel.pth'")

# Evaluation on the test dataset
print("\nEvaluating on test data...")
model.eval()
with torch.no_grad():
    start_time = time.time()
    outputs = model(X_test_tensor)
    _, preds = torch.max(outputs, 1)
    end_time = time.time()

# Convert to NumPy
y_pred_np = preds.numpy()
y_true_np = y_test_tensor.numpy()

# Compute confusion matrix values
TP = np.sum((y_pred_np == 1) & (y_true_np == 1))
TN = np.sum((y_pred_np == 0) & (y_true_np == 0))
FP = np.sum((y_pred_np == 1) & (y_true_np == 0))
FN = np.sum((y_pred_np == 0) & (y_true_np == 1))

# Compute metrics manually
def compute_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn + 1e-8)

def compute_precision(tp, fp):
    return tp / (tp + fp + 1e-8)

def compute_recall(tp, fn):
    return tp / (tp + fn + 1e-8)

def compute_f1(precision, recall):
    return 2 * precision * recall / (precision + recall + 1e-8)

accuracy = compute_accuracy(TP, TN, FP, FN)
precision = compute_precision(TP, FP)
recall = compute_recall(TP, FN)
f1 = compute_f1(precision, recall)

# Print evaluation results
print(f"\nInference Time: {(end_time - start_time):.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Now testing with another dataset (e.g., KDDTest.txt)

print("\nLoading test dataset...")
X_test_new, y_test_new = preprocess_data("../data/KDDTest+.txt")

# Convert to PyTorch tensors
X_test_new_tensor = torch.tensor(X_test_new)
y_test_new_tensor = torch.tensor(y_test_new)

# Evaluation on the new test dataset
print("\nEvaluating on new test data...")
model.eval()
with torch.no_grad():
    start_time = time.time()
    outputs = model(X_test_new_tensor)
    _, preds = torch.max(outputs, 1)
    end_time = time.time()

# Convert to NumPy
y_pred_new_np = preds.numpy()
y_true_new_np = y_test_new_tensor.numpy()

# Compute confusion matrix values
TP_new = np.sum((y_pred_new_np == 1) & (y_true_new_np == 1))
TN_new = np.sum((y_pred_new_np == 0) & (y_true_new_np == 0))
FP_new = np.sum((y_pred_new_np == 1) & (y_true_new_np == 0))
FN_new = np.sum((y_pred_new_np == 0) & (y_true_new_np == 1))

# Compute metrics manually for the new dataset
accuracy_new = compute_accuracy(TP_new, TN_new, FP_new, FN_new)
precision_new = compute_precision(TP_new, FP_new)
recall_new = compute_recall(TP_new, FN_new)
f1_new = compute_f1(precision_new, recall_new)

# Print evaluation results for the new dataset
print(f"\nInference Time: {(end_time - start_time):.6f} seconds")
print(f"Accuracy (new dataset): {accuracy_new:.4f}")
print(f"Precision (new dataset): {precision_new:.4f}")
print(f"Recall (new dataset): {recall_new:.4f}")
print(f"F1 Score (new dataset): {f1_new:.4f}")
print("\nConfusion Matrix (new dataset):")
print(f"TP: {TP_new}, TN: {TN_new}, FP: {FP_new}, FN: {FN_new}")
