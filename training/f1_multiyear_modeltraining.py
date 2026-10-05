import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import json
import shap

# ── Load CSV ──────────────────────────────────────────────────
df = pd.read_csv('f1_multiyear.csv')
print("Loaded:", df.shape)

# ── Create target variable ────────────────────────────────────
df['Winner'] = (df['Position'] == 1).astype(int)

# ── Drop rows with missing values ─────────────────────────────
df = df.dropna(subset=[
    'GridPosition', 'QualiPosition',
    'Rainfall', 'AirTemp', 'TrackTemp',
    'StandingPoints', 'HomeRace'
])

# ── Label Encoding ────────────────────────────────────────────
le_driver = LabelEncoder()
le_team = LabelEncoder()
le_circuit = LabelEncoder()

df['Driver'] = le_driver.fit_transform(df['Abbreviation'])
df['Team'] = le_team.fit_transform(df['TeamName'])
df['Circuit'] = le_circuit.fit_transform(df['Race'])

# ── Features & Target ─────────────────────────────────────────
feature_names = ['Driver', 'Team', 'Circuit', 'GridPosition', 'QualiPosition',
                  'Rainfall', 'AirTemp', 'TrackTemp', 'StandingPoints', 'HomeRace']
X = df[feature_names]
y = df['Winner']

print("Class distribution:")
print(y.value_counts())

# ── Train/Test Split FIRST (raw, un-resampled, un-scaled) ─────
# Splitting before SMOTE prevents synthetic points derived from
# training rows leaking neighbor information into the test set.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE — fit ONLY on the training fold ──────────────────────
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_raw, y_train)
print("\nTrain class balance after SMOTE:", y_train_res.value_counts())
print("Test set (untouched, real distribution):", y_test.value_counts())

# ── Scaling — fit on train, apply to both ──────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test_raw)

# ── Train XGBoost ─────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb.fit(X_train_scaled, y_train_res)

train_acc = xgb.score(X_train_scaled, y_train_res)
test_acc = xgb.score(X_test_scaled, y_test)

print("\nTrain Accuracy (SMOTE-balanced train fold):", round(train_acc * 100, 2), "%")
print("Test Accuracy (real, untouched distribution):", round(test_acc * 100, 2), "%")
print("\nClassification Report (test set):")
print(classification_report(y_test, xgb.predict(X_test_scaled)))

# ── Save model + encoders + scaler ───────────────────────────
with open('f1_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)

with open('f1_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('f1_encoders.pkl', 'wb') as f:
    pickle.dump({
        'driver': le_driver,
        'team': le_team,
        'circuit': le_circuit
    }, f)

print("\n✅ Model, scaler and encoders saved!")

# ── Feature importance (built-in XGBoost) ──────────────────────
importance_dict = dict(zip(feature_names, xgb.feature_importances_.tolist()))
with open('f1_feature_importance.json', 'w') as f:
    json.dump(importance_dict, f)
print("✅ Feature importance saved!")

# ── SHAP values — MUST use scaled input, same as training ─────
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(xgb)
shap_sample = X_test_scaled[:100]  # scaled, matches what the model was trained on
shap_values = explainer.shap_values(shap_sample)

mean_shap = np.abs(shap_values).mean(axis=0).tolist()
shap_importance = dict(zip(feature_names, mean_shap))
with open('f1_shap_values.json', 'w') as f:
    json.dump(shap_importance, f)

print("✅ SHAP values saved!")
print("\nFeature importance via SHAP:")
for feat, val in sorted(shap_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {val:.4f}")

# ── Real metrics, computed on the correct scaled splits ────────
metrics = {
    'train_accuracy': round(train_acc * 100, 2),
    'test_accuracy': round(test_acc * 100, 2),
    'n_samples': len(df),
    'seasons': '2019-2025'
}
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"✅ Metrics saved: {metrics}")