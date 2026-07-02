import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import pickle

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = r'D:\NIDS_Project\data'
MODEL_DIR = r'D:\NIDS_Project\models'
PLOTS_DIR = r'D:\NIDS_Project\plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Load model ─────────────────────────────────────────────────────────────
print("Loading model...")
model = xgb.XGBClassifier()
model.load_model(os.path.join(MODEL_DIR, 'xgboost_ids_model.json'))
print("✅ Model loaded!")

# ── 2. Load test data ─────────────────────────────────────────────────────────
print("\nLoading test data...")
X_test = pd.read_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
y_test = pd.read_parquet(os.path.join(DATA_DIR, 'y_test.parquet'))['Attack_label']
print(f"✅ Test data loaded: {X_test.shape}")

# ── 3. Re-apply encoding (same as train.py) ───────────────────────────────────
print("\nEncoding categorical columns...")
from sklearn.preprocessing import LabelEncoder

X_train = pd.read_parquet(os.path.join(DATA_DIR, 'X_train.parquet'))
X_val   = pd.read_parquet(os.path.join(DATA_DIR, 'X_val.parquet'))

str_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()

label_encoders = {}
for col in str_cols:
    n_unique_train = X_train[col].nunique()
    if n_unique_train <= 50:
        le = LabelEncoder()
        all_vals = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str).unique()
        le.fit(all_vals)
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    else:
        X_test[col] = X_test[col].astype(str).apply(lambda x: hash(x) % 100000)

print("✅ Encoding complete!")

# ── 4. Sample for SHAP (use 1000 rows — full test set too slow) ───────────────
print("\nSampling 1000 rows for SHAP analysis...")
sample_idx    = np.random.choice(len(X_test), size=1000, replace=False)
X_sample      = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample      = y_test.iloc[sample_idx].reset_index(drop=True)
print(f"✅ Sample: {X_sample.shape} | Attacks: {y_sample.sum()} | Normal: {(y_sample==0).sum()}")

# ── 5. Compute SHAP values ────────────────────────────────────────────────────
print("\nComputing SHAP values (this may take 1-2 minutes)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
print("✅ SHAP values computed!")

# ── 6. Save SHAP values for dashboard use ────────────────────────────────────
print("\nSaving SHAP data...")
with open(os.path.join(MODEL_DIR, 'shap_explainer.pkl'), 'wb') as f:
    pickle.dump(explainer, f)

np.save(os.path.join(MODEL_DIR, 'shap_values.npy'), shap_values.values)
X_sample.to_parquet(os.path.join(MODEL_DIR, 'shap_sample.parquet'))
y_sample.to_frame().to_parquet(os.path.join(MODEL_DIR, 'shap_sample_labels.parquet'))
print("✅ SHAP data saved!")

# ── 7. Global Summary Plot (top features overall) ────────────────────────────
print("\nGenerating SHAP Summary Plot...")
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_sample,
    plot_type = 'bar',
    max_display = 20,
    show = False
)
plt.title('Top 20 Features — Global SHAP Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ SHAP summary bar plot saved!")

# ── 8. SHAP Beeswarm Plot (feature impact direction) ─────────────────────────
print("Generating SHAP Beeswarm Plot...")
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_sample,
    max_display = 20,
    show = False
)
plt.title('SHAP Beeswarm — Feature Impact on Attack Detection')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ SHAP beeswarm plot saved!")

# ── 9. Single Prediction Explanation (waterfall for 1 attack sample) ──────────
print("Generating SHAP Waterfall Plot (single attack explanation)...")
attack_indices = np.where(y_sample == 1)[0]
if len(attack_indices) > 0:
    idx = attack_indices[0]
    plt.figure()
    shap.waterfall_plot(shap_values[idx], max_display=15, show=False)
    plt.title('Why This Packet Was Flagged as Attack')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_waterfall_attack.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP waterfall plot saved!")

# ── 10. Print top features per prediction ─────────────────────────────────────
print("\n" + "=" * 55)
print("TOP 5 FEATURES DRIVING ATTACK DETECTION")
print("=" * 55)
mean_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0),
    index=X_sample.columns
).sort_values(ascending=False)

for i, (feat, val) in enumerate(mean_shap.head(5).items()):
    print(f"  {i+1}. {feat:<35} SHAP: {val:.4f}")

print("\n✅ All SHAP analysis complete!")
print(f"📁 Plots saved to: {PLOTS_DIR}")
print(f"📁 SHAP data saved to: {MODEL_DIR}")