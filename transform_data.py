from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class DataTransformer:

    def encode_data(self, df):

        print("\nStarting Data Processing...")

        # Remove Description Column
        df.drop(
            columns=['description',
                     'inventoryid',
                     'vendorname',
                     'approval'],
            inplace=True,
            errors='ignore'
        )

        # Clean Size Column
        if 'size' in df.columns:

            print("\nCleaning Size Column...")

            df['size'] = (
                df['size']
                .astype(str)
                .str.lower()
                .str.strip()
            )

            # Convert Liter to 1000
            df['size'] = df['size'].replace({
                'liter': '1000',
                '1l': '1000'
            })

            # Remove ml
            df['size'] = df['size'].str.replace(
                'ml',
                '',
                regex=False
            )

            # Remove spaces
            df['size'] = df['size'].str.replace(
                ' ',
                '',
                regex=False
            )

            # Convert to numeric
            df['size'] = pd.to_numeric(
                df['size'],
                errors='coerce'
            )

            # Fill missing values
            df['size'] = df['size'].fillna(0)

            print("Size Column Converted Successfully")

        # Encode City Only
        if 'city' in df.columns:

            le_city = LabelEncoder()

            df['city'] = le_city.fit_transform(
                df['city'].astype(str)
            )

            print("City Encoded Successfully")

        print("\nData Processing Completed")

        return df


# =========================================
# LOAD DATASET
# =========================================

print("\nLoading Dataset...")

df = pd.read_csv(
    r"C:\Users\nehas\OneDrive\Desktop\Vendor_Intelligence_System\data\raw\sales.csv",
    nrows=500000,
    low_memory=False
)

# =========================================
# CLEAN COLUMN NAMES
# =========================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# =========================================
# CONVERT DATE COLUMNS
# =========================================

date_cols = [
    col for col in df.columns
    if 'date' in col.lower()
]

for col in date_cols:

    print(f"\nConverting Date Column: {col}")

    df[col] = pd.to_datetime(
        df[col],
        errors='coerce'
    )

    df[col] = (
        df[col].astype('int64') // 10**9
    )

    df[col] = df[col].replace(
        -9223372037,
        0
    )

# =========================================
# CREATE TARGET COLUMN
# =========================================

if 'salesdollars' in df.columns:

    print("\nCreating Target Column...")

    df['high_sales'] = (
        df['salesdollars']
        > df['salesdollars'].mean()
    ).astype('int8')

# =========================================
# TRANSFORM DATA
# =========================================

transformer = DataTransformer()

df = transformer.encode_data(df)

# =========================================
# HANDLE MISSING VALUES
# =========================================

print("\nHandling Missing Values...")

numeric_cols = df.select_dtypes(
    include=np.number
).columns

df[numeric_cols] = df[numeric_cols].fillna(0)

# =========================================
# OPTIMIZE MEMORY
# =========================================

print("\nOptimizing Memory...")

for col in numeric_cols:

    if df[col].dtype == 'float64':

        df[col] = pd.to_numeric(
            df[col],
            downcast='float'
        )

    elif df[col].dtype == 'int64':

        df[col] = pd.to_numeric(
            df[col],
            downcast='integer'
        )

# =========================================
# SAVE PROCESSED DATASET
# =========================================

print("\nSaving Processed Dataset...")

df.to_csv(
    "sales_processed_500k.csv",
    index=False
)

# =========================================
# FINAL OUTPUT
# =========================================

print("\nProcessed Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMemory Usage:")

print(
    round(
        df.memory_usage(deep=True).sum() / 1024**2,
        2
    ),
    "MB"
)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nData Transformation Completed Successfully!")