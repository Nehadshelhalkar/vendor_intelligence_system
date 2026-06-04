import pandas as pd
import numpy as np

class DataCleaner:

    # =====================================
    # REMOVE DUPLICATES
    # =====================================

    def remove_duplicates(self, df):

        print("\n========== DUPLICATE CHECK ==========")

        before_rows = df.shape[0]

        print(
            f"\nRows Before Removing Duplicates: "
            f"{before_rows}"
        )

        duplicate_count = df.duplicated().sum()

        print(
            f"Duplicate Rows Found: "
            f"{duplicate_count}"
        )

        df = df.drop_duplicates()

        after_rows = df.shape[0]

        print(
            f"Rows After Removing Duplicates: "
            f"{after_rows}"
        )

        return df

    # =====================================
    # HANDLE MISSING VALUES
    # =====================================

    def handle_missing_values(self, df):

        print("\n========== NULL VALUE CHECK ==========")

        # BEFORE CLEANING
        print("\nMissing Values BEFORE Cleaning:\n")

        print(df.isnull().sum())

        print(
            f"\nTotal Null Values Before: "
            f"{df.isnull().sum().sum()}"
        )

        # =====================================
        # NUMERICAL COLUMNS
        # =====================================

        num_cols = df.select_dtypes(
            include=np.number
        ).columns

        # =====================================
        # CATEGORICAL COLUMNS
        # =====================================

        cat_cols = df.select_dtypes(
            include='object'
        ).columns

        # Fill Numerical Nulls
        for col in num_cols:

            df[col] = df[col].fillna(
                df[col].mean()
            )

        # Fill Categorical Nulls
        for col in cat_cols:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

        # AFTER CLEANING
        print("\nMissing Values AFTER Cleaning:\n")

        print(df.isnull().sum())

        print(
            f"\nTotal Null Values After: "
            f"{df.isnull().sum().sum()}"
        )

        print("\nMissing Values Handled")

        return df

    # =====================================
    # STANDARDIZE COLUMNS
    # =====================================

    def standardize_columns(self, df):

        print("\n========== COLUMN STANDARDIZATION ==========")

        print("\nColumns BEFORE:\n")

        print(df.columns.tolist())

        df.columns = (
            df.columns
            .str.lower()
            .str.replace(" ", "_")
        )

        print("\nColumns AFTER:\n")

        print(df.columns.tolist())

        print("\nColumns Standardized")

        return df