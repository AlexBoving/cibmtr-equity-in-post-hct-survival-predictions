import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import spearmanr
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import csv

# Load the datasets
train_path = r"data\train.csv"
test_path = r"data\test.csv"
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Drop ID column since it's not useful for prediction
df_train.drop(columns=['ID'], inplace=True, errors='ignore')
df_test.drop(columns=['ID'], inplace=True, errors='ignore')

# Define target and time columns
target_col = 'efs'
time_col = 'efs_time'

# Separate features and target
y_train = df_train[target_col]
time_train = df_train[time_col]
X_train = df_train.drop(columns=[target_col, time_col])
X_test = df_test.drop(columns=[time_col], errors='ignore')  # Ensure it exists

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Fill missing values
X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].median())
X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train[numerical_cols].median())
X_train[categorical_cols] = X_train[categorical_cols].fillna('Unknown')
X_test[categorical_cols] = X_test[categorical_cols].fillna('Unknown')

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Split training data into train-validation sets
X_train_split, X_val, y_train_split, y_val, time_train_split, time_val = train_test_split(
    X_train, y_train, time_train, test_size=0.2, random_state=42, stratify=y_train
)

# Define and train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train_split, y_train_split)

# Predict probabilities for validation set
y_val_pred = xgb_model.predict_proba(X_val)[:, 1]

# Compute Spearman's rank correlation as a proxy for C-index
#spearman_corr, _ = spearmanr(time_val, y_val_pred)

# Output Spearman's correlation
#print("Spearman's Correlation (Proxy for C-index):", spearman_corr)

#C-index
cindex = concordance_index(time_val, y_val_pred)

#c-index output
print("The unstratified c-index is calculated as:", cindex )

# Predict on test set
y_test_pred = xgb_model.predict_proba(X_test)[:, 1]
print("Test Predictions:", y_test_pred)

# Log metrics to CSV
log_filename = "model_metrics_tracker.csv"
headers = ["model used", "Imputation used", "number of features", "features removed", "C1 Score", "Test predictions"]
data = [
    "XGBoost", "Median for numerical, 'Unknown' for categorical", X_train.shape[1], "None", cindex, list(y_test_pred)
]

with open(log_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerow(data)

print(f"Metrics logged to {log_filename}")

# Perform T-SNE transformation
#tsne = TSNE(n_components=2, random_state=42)
#X_embedded = tsne.fit_transform(X_train)

# Plot T-SNE
#plt.figure(figsize=(10, 6))
#sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_train, palette='coolwarm', alpha=0.7)
#plt.title("T-SNE Visualization of Data")
#plt.xlabel("T-SNE Component 1")
#plt.ylabel("T-SNE Component 2")
#plt.legend(title='EFS')
#plt.show()
