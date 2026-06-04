class DataValidator:

    def validate_dataframe(self, df):

        print("\n==============================")
        print("DATA VALIDATION")
        print("==============================")

        print("\nDataset Shape:")

        print(df.shape)

        print("\nColumn Names:")

        print(df.columns.tolist())

        print("\nMissing Values:")

        print(df.isnull().sum())

        print("\nDuplicate Rows:")

        print(df.duplicated().sum())

        return True