"""
Historical Explain-Health random forest trainer and asset exporter.
The original row-level split shares symptom profiles across training and test.
Use reproducibility/audit.py for the deduplicated benchmark and overlap audit.
This script writes model and precaution assets in the current directory.
Requirements: pip install scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
import json
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. LOAD AND PARSE DATASET
# ─────────────────────────────────────────────
df = pd.read_csv('DiseaseAndSymptoms.csv')

# Collect all unique symptoms (131 total)
all_symptoms = set()
for col in df.columns[1:]:
    for val in df[col].dropna():
        s = str(val).strip().replace('_', ' ').lower()
        if s and s != 'null':
            all_symptoms.add(s)

symptom_list = sorted(list(all_symptoms))
symptom_index = {s: i for i, s in enumerate(symptom_list)}

print(f"Total unique symptoms: {len(symptom_list)}")

# ─────────────────────────────────────────────
# 2. BUILD FEATURE MATRIX
# Each row = binary vector of symptoms present/absent
# ─────────────────────────────────────────────
X = []  # feature vectors
y = []  # disease labels

for _, row in df.iterrows():
    disease = str(row.iloc[0]).strip()
    if not disease:
        continue

    feature_vector = [0] * len(symptom_list)

    for col in df.columns[1:]:
        val = str(row[col]).strip().replace('_', ' ').lower()
        if val and val != 'nan' and val != 'null' and val in symptom_index:
            feature_vector[symptom_index[val]] = 1

    if sum(feature_vector) > 0:
        X.append(feature_vector)
        y.append(disease)

X = np.array(X)
y = np.array(y)

disease_list = sorted(list(set(y)))
disease_index = {d: i for i, d in enumerate(disease_list)}
y_encoded = np.array([disease_index[d] for d in y])

print(f"Eligible dataset rows before splitting: {len(X)}")
print(f"Unique diseases: {len(disease_list)}")

# ─────────────────────────────────────────────
# 3. TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 4. REPRODUCE THE HISTORICAL ROW-SPLIT METRIC
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nHistorical row-split accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print("Interpret this as agreement on a split with overlapping symptom profiles.")
print("See reproducibility/README.md for the deduplicated evaluation.")

# ─────────────────────────────────────────────
# 5. EXPORT MODEL AS JSON (for Android)
# No TFLite needed - we export the tree structure
# Android reads this JSON and does inference itself
# ─────────────────────────────────────────────

# Export each tree as simplified rules (top predictions per leaf)
def export_forest(forest, symptom_list, disease_list):
    """Export Random Forest as a list of decision trees in JSON"""
    trees = []
    
    for estimator in forest.estimators_:
        tree = estimator.tree_
        
        def export_node(node_id):
            if tree.children_left[node_id] == -1:
                # Leaf node - get class probabilities
                values = tree.value[node_id][0]
                total = sum(values)
                probs = [float(v/total) for v in values]
                return {"leaf": True, "probs": probs}
            else:
                return {
                    "leaf": False,
                    "feature": int(tree.feature[node_id]),
                    "left": export_node(tree.children_left[node_id]),
                    "right": export_node(tree.children_right[node_id])
                }
        
        trees.append(export_node(0))
    
    return trees

print("\nExporting model to JSON...")
trees = export_forest(model, symptom_list, disease_list)

model_data = {
    # Preserve the field for compatibility with the Android reader.
    "accuracy": float(accuracy),
    "n_trees": len(trees),
    "symptoms": symptom_list,
    "diseases": disease_list,
    "trees": trees
}

with open('rf_model.json', 'w') as f:
    json.dump(model_data, f, separators=(',', ':'))  # compact JSON

import os
size_kb = os.path.getsize('rf_model.json') / 1024
print(f"✅ Saved rf_model.json ({size_kb:.0f} KB)")

# ─────────────────────────────────────────────
# 6. EXPORT PRECAUTIONS AS JSON
# ─────────────────────────────────────────────
precautions = {}
with open('Disease_precaution.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 2: continue
        disease = row[0].strip()
        precs = [p.strip() for p in row[1:] if p.strip() and p.strip() != 'null']
        if disease and precs:
            precautions[disease] = precs

with open('precautions.json', 'w') as f:
    json.dump(precautions, f, indent=2)

print("✅ Saved precautions.json")

print("\n📦 FILES TO COPY INTO ANDROID app/src/main/assets/:")
print("   - rf_model.json")
print("   - precautions.json")
print("\nDONE!")
