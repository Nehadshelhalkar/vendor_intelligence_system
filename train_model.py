import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from xgboost import XGBClassifier

# =========================================
# LOAD DATASET
# =========================================

print("Loading Processed Dataset...")

df = pd.read_csv(
    "data/processed/vendor_sales_summary_processed.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print(
    "\nDuplicate Rows:",
    df.duplicated().sum()
)

df = df.drop_duplicates()

# =========================================
# CREATE HIGH SALES TARGET
# =========================================

print("\nCreating High Sales Target...")


threshold = (
    df['totalsalesdollars']
    .median()
)

df['high_sales'] = (
    df['totalsalesdollars']
    > threshold
).astype(int)

print("\nTarget Distribution:")
print(df['high_sales'].value_counts())

# =========================================
# REMOVE OBJECT COLUMNS
# =========================================

object_cols = df.select_dtypes(
    include='object'
).columns

if len(object_cols) > 0:

    print("\nRemoving Object Columns:")
    print(object_cols.tolist())

    df.drop(
        columns=object_cols,
        inplace=True
    )

# =========================================
# FEATURES & TARGET
# ========================================
X = df.drop(
    columns=[
        'high_sales',
        'totalsalesdollars',
        'grossprofit',
        'profitmargin',
        'totalsalesprice',
        'totalsalesquantity',
        'salestopurchaseratio'
    ],
    errors='ignore'
)
y = df['high_sales']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)
print(X.columns)

# =========================================
# CLEAN NUMERIC DATA
# =========================================
import numpy as np

print("\nChecking INF values...")

for col in X.columns:

    inf_count = np.isinf(X[col]).sum()

    if inf_count > 0:

        print(
            f"{col}: {inf_count} INF values"
        )
        
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

# Cap extremely large values
X = X.clip(
    lower=-1000000,
    upper=1000000
)

print("\nInfinity Values Removed")

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =========================================
# XGBOOST MODEL
# =========================================

scale_pos_weight = (
    y.value_counts()[0]
    /
    y.value_counts()[1]
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
# =========================================
# TRAIN MODEL
# =========================================

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

print("\nTraining Completed")

# =========================================
# TRAIN ACCURACY
# =========================================

train_pred = model.predict(
    X_train
)

train_acc = accuracy_score(
    y_train,
    train_pred
)

print(
    "\nTrain Accuracy:",
    round(train_acc * 100, 2),
    "%"
)

# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================================
# ACCURACY
# =========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================
# CONFUSION MATRIX
# =========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
auc = roc_auc_score(
    y_test,
    y_prob
)

print(
    "\nROC-AUC Score:",
    round(auc, 4)
)
# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:")

print(
    importance.head(10)
)
importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print(
    "\nFeature Importance Saved"
)

# =========================================
# SAVE MODEL
# =========================================

joblib.dump(
    model,
    "outputs/vendor_xgboost_model.pkl"
)

print(
    "\nModel Saved Successfully"
)

print(
    "Location: outputs/vendor_xgboost_model.pkl"
)
# =========================================
# CROSS VALIDATION
# =========================================

print("\nRunning Cross Validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring='roc_auc'
)

print(scores)
print(scores.mean())
