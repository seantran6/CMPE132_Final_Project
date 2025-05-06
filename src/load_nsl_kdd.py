import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Path to your data folder
DATA_PATH = 'data/'

# Load field names from CSV
field_names = pd.read_csv(os.path.join(DATA_PATH, 'Field_Names.csv'), header=None)[0].tolist()
field_names.append('label')  # Add the class label

# Load training and test datasets
train_df = pd.read_csv(os.path.join(DATA_PATH, 'KDDTrain+.TXT'), names=field_names)
test_df = pd.read_csv(os.path.join(DATA_PATH, 'KDDTest+.TXT'), names=field_names)

# Display dataset shapes
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Preview the dataset
print(train_df.head())

# STEP 1: Preprocessing

# 1.1. Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 1.2. Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()

for col in categorical_cols:
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])

# 1.3. Normalize numerical features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 2: Encode labels (Binary classification: 'normal' vs 'attack')
y_train = y_train.apply(lambda x: 0 if x == 'normal' else 1)
y_test = y_test.apply(lambda x: 0 if x == 'normal' else 1)

# STEP 3: Train a baseline RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nClassification Report (RandomForest):")
print(classification_report(y_test, y_pred))
