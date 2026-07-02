import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR  = r'D:\NIDS_Project\data'
MODEL_DIR = r'D:\NIDS_Project\models'
PLOTS_DIR = r'D:\NIDS_Project\plots'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_parquet(r'D:\NIDS_Project\data\X_train.parquet')
X_val   = pd.read_parquet(r'D:\NIDS_Project\data\X_val.parquet')
X_test  = pd.read_parquet(r'D:\NIDS_Project\data\X_test.parquet')
y_train = pd.read_parquet(r'D:\NIDS_Project\data\y_train.parquet')['Attack_label']
y_val   = pd.read_parquet(r'D:\NIDS_Project\data\y_val.parquet')['Attack_label']
y_test  = pd.read_parquet(r'D:\NIDS_Project\data\y_test.parquet')['Attack_label']
print(f"✅ Data loaded — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── 2. Encode categorical columns (hash trick for high-cardinality) ────────────
print("\nEncoding categorical columns...")

# Check which columns are text/string type
str_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()
print(f"Text columns found ({len(str_cols)}): {str_cols}")

for col in str_cols:
    n_unique_train = X_train[col].nunique()
    print(f"  {col:<35} unique values: {n_unique_train}")

    if n_unique_train <= 50:
        # Low cardinality → safe to use LabelEncoder
        le = LabelEncoder()
        all_vals = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str).unique()
        le.fit(all_vals)  # fit on ALL data to avoid unseen label error
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col]   = le.transform(X_val[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
        print(f"    → LabelEncoded")
    else:
        # High cardinality (like HTML content) → use hash trick
        X_train[col] = X_train[col].astype(str).apply(lambda x: hash(x) % 100000)
        X_val[col]   = X_val[col].astype(str).apply(lambda x: hash(x) % 100000)
        X_test[col]  = X_test[col].astype(str).apply(lambda x: hash(x) % 100000)
        print(f"    → Hash encoded (high cardinality)")

print("✅ Encoding complete!")

# ── 3. Train XGBoost ──────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators     = 100,
    max_depth        = 6,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = 'logloss',
    random_state     = 42,
    n_jobs           = -1
)

model.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)],
    verbose  = 10
)
print("✅ Training complete!")

# ── 4. Evaluate on test set ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TEST SET EVALUATION")
print("=" * 55)
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# ── 5. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix — XGBoost Binary IDS')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()
print("✅ Confusion matrix saved!")

# ── 6. Feature Importance ─────────────────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
importance.plot(kind='barh', color='steelblue')
plt.title('Top 20 Most Important Features — XGBoost')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150)
plt.show()
print("✅ Feature importance saved!")

# ── 7. Save model ─────────────────────────────────────────────────────────────
model.save_model(os.path.join(MODEL_DIR, 'xgboost_ids_model.json'))
print("✅ Model saved!")