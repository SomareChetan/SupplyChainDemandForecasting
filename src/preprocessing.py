import pandas as pd

# Load datasets
print("=" * 60)
print("Loading Walmart Datasets...")
print("=" * 60)

train = pd.read_csv("data/raw/train.csv")
features = pd.read_csv("data/raw/features.csv")
stores = pd.read_csv("data/raw/stores.csv")

print("\n✅ Datasets Loaded Successfully!")

# Data audit
datasets = {
    "TRAIN": train,
    "FEATURES": features,
    "STORES": stores
}

for name, df in datasets.items():

    print("\n" + "=" * 60)
    print(f"{name} DATASET")
    print("=" * 60)

    print(f"\nShape : {df.shape}")

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

# Data cleaning
print("\n" + "=" * 60)
print("Starting Data Cleaning...")
print("=" * 60)

# Convert Date columns
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

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

print("✅ Data Cleaning Completed!")

# Merge datasets
print("\n" + "=" * 60)
print("Merging Datasets...")
print("=" * 60)

merged_data = pd.merge(
    train,
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

merged_data = pd.merge(
    merged_data,
    stores,
    on="Store",
    how="left"
)

print("✅ Merge Completed!")

# Feature engineering
print("\n" + "=" * 60)
print("Performing Feature Engineering...")
print("=" * 60)

merged_data["Year"] = merged_data["Date"].dt.year
merged_data["Month"] = merged_data["Date"].dt.month
merged_data["Month_Name"] = merged_data["Date"].dt.month_name()
merged_data["Quarter"] = merged_data["Date"].dt.quarter
merged_data["Week"] = merged_data["Date"].dt.isocalendar().week.astype(int)
merged_data["Day"] = merged_data["Date"].dt.day
merged_data["Day_Name"] = merged_data["Date"].dt.day_name()

print("✅ Feature Engineering Completed!")

print("\nNew Columns Added:")
print([
    "Year",
    "Month",
    "Month_Name",
    "Quarter",
    "Week",
    "Day",
    "Day_Name"
])

# Save processed dataset
output_path = "data/processed/cleaned_data.csv"

merged_data.to_csv(
    output_path,
    index=False
)

print("\n✅ Dataset Saved Successfully!")
print(f"\nLocation : {output_path}")

print("\nFinal Dataset Shape:")
print(merged_data.shape)

print("\nFirst Five Rows:")
print(merged_data.head())