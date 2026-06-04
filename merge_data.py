class DataMerger:

    def merge_datasets(self, datasets):

        print("\nStarting Safe Dataset Merge...")

        # Base Dataset
        master_df = datasets["sales"]

        print("\nInitial Shape:")
        print(master_df.shape)

        # =================================================
        # Merge purchase_prices
        # =================================================

        if "purchase_prices" in datasets:

            print("\nMerging purchase_prices...")

            purchase_prices = datasets[
                "purchase_prices"
            ]

            # Keep only required columns
            required_cols = [
                "brand",
                "price"
            ]

            available_cols = [
                col for col in required_cols
                if col in purchase_prices.columns
            ]

            purchase_prices = purchase_prices[
                available_cols
            ]

            if "brand" in master_df.columns and \
               "brand" in purchase_prices.columns:

                master_df = master_df.merge(
                    purchase_prices,
                    on="brand",
                    how="left",
                    suffixes=("", "_purchase")
                )

                print(
                    "\nAfter purchase_prices merge:"
                )

                print(master_df.shape)

        # =================================================
        # Merge vendor_sales_summary
        # =================================================

        if "vendor_sales_summary" in datasets:

            print("\nMerging vendor_sales_summary...")

            vendor_summary = datasets[
                "vendor_sales_summary"
            ]

            required_cols = [
                "vendorno",
                "salesdollars",
                "salesquantity"
            ]

            available_cols = [
                col for col in required_cols
                if col in vendor_summary.columns
            ]

            vendor_summary = vendor_summary[
                available_cols
            ]

            if "vendorno" in master_df.columns and \
               "vendorno" in vendor_summary.columns:

                master_df = master_df.merge(
                    vendor_summary,
                    on="vendorno",
                    how="left",
                    suffixes=("", "_summary")
                )

                print(
                    "\nAfter vendor_sales_summary merge:"
                )

                print(master_df.shape)

        # =================================================
        # Merge vendor_invoice
        # =================================================

        if "vendor_invoice" in datasets:

            print("\nMerging vendor_invoice...")

            vendor_invoice = datasets[
                "vendor_invoice"
            ]

            required_cols = [
                "vendornumber",
                "freight",
                "dollars"
            ]

            available_cols = [
                col for col in required_cols
                if col in vendor_invoice.columns
            ]

            vendor_invoice = vendor_invoice[
                available_cols
            ]

            # Remove duplicate vendor numbers
            if "vendornumber" in vendor_invoice.columns:

                vendor_invoice = (
                    vendor_invoice
                    .drop_duplicates(
                        subset=["vendornumber"]
                    )
                )

            if "vendorno" in master_df.columns and \
               "vendornumber" in vendor_invoice.columns:

                master_df = master_df.merge(
                    vendor_invoice,
                    left_on="vendorno",
                    right_on="vendornumber",
                    how="left",
                    suffixes=("", "_invoice")
                )

                print(
                    "\nAfter vendor_invoice merge:"
                )

                print(master_df.shape)

        print("\nDataset Merge Completed")

        return master_df