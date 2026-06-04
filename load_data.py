import pandas as pd
import os

class DataLoader:

    def __init__(self):

        self.raw_data_path = "data/raw"

    def load_csv_files(self):

        datasets = {}

        files = os.listdir(self.raw_data_path)

        for file in files:

            if file.endswith(".csv"):

                file_path = os.path.join(
                    self.raw_data_path,
                    file
                )

                print(f"\nLoading File: {file}")

                try:

                    # =====================================
                    # LOAD ONLY 500K ROWS FOR SALES DATA
                    # =====================================

                    if file == "sales.csv":

                        print(
                            "\nLoading only 500,000 rows "
                            "from sales.csv"
                        )

                        df = pd.read_csv(
                            file_path,
                            encoding='latin1',
                            nrows=500000
                        )

                    # =====================================
                    # LOAD FULL DATA FOR OTHER FILES
                    # =====================================

                    else:

                        df = pd.read_csv(
                            file_path,
                            encoding='latin1'
                        )

                    dataset_name = file.replace(
                        ".csv",
                        ""
                    )

                    datasets[dataset_name] = df

                    print(
                        f"{dataset_name} Loaded Successfully"
                    )

                    print(df.shape)

                except Exception as e:

                    print(
                        f"Error Loading {file}: {e}"
                    )

        return datasets