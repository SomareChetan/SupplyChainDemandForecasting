# =====================================================
# SUPPLY CHAIN DEMAND FORECASTING PROJECT
# Person A - Data Engineering Pipeline
# =====================================================

# ============================
# 1. Import Libraries
# ============================

import pandas as pd

# ============================
# 2. Load Datasets
# ============================

print("=" * 60)
print("Loading Walmart Datasets...")
print("=" * 60)

train = pd.read_csv("data/raw/train.csv")
features = pd.read_csv("data/raw/features.csv")
stores = pd.read_csv("data/raw/stores.csv")

print("\n✅ Datasets Loaded Successfully!")

# ============================
# 3. Data Audit
# ============================

datasets = {
    "TRAIN": train,
    "FEATURES": features,
    "STORES": stores
}

for name, df in datasets.items():

    print("\n" + "=" * 60)
    print(f"{name} DATASET")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

# ============================
# 4. Data Cleaning
# ============================

print("\n" + "=" * 60)
print("Starting Data Cleaning...")
print("=" * 60)

# Convert Date columns
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

print("\n✅ Date columns converted successfully.")

# Fill missing MarkDown values
markdown_columns = [
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5"
]

features[markdown_columns] = features[markdown_columns].fillna(0)

# Fill missing CPI and Unemployment values
features["CPI"] = features["CPI"].ffill().bfill()
features["Unemployment"] = features["Unemployment"].ffill().bfill()

print("✅ Missing values handled.")

print("\nRemaining Missing Values")
print(features.isnull().sum())

print("\n✅ Data Cleaning Completed Successfully!")

# ============================
# 5. Merge Datasets
# ============================

print("\n" + "=" * 60)
print("Merging Datasets...")
print("=" * 60)

# Merge train and features
merged_data = pd.merge(
    train,
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

# Merge with stores
merged_data = pd.merge(
    merged_data,
    stores,
    on="Store",
    how="left"
)

print("✅ Datasets merged successfully!")

print("\nMerged Dataset Shape:")
print(merged_data.shape)

print("\nMerged Dataset Columns:")
print(merged_data.columns.tolist())

print("\nFirst 5 Rows:")
print(merged_data.head())

# ============================
# 6. Save Processed Dataset
# ============================

output_path = "data/processed/cleaned_data.csv"

merged_data.to_csv(output_path, index=False)

print(f"\n✅ Cleaned dataset saved successfully!")
print(f"Location: {output_path}")