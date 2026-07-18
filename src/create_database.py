import sqlite3
import pandas as pd

print("=" * 60)
print("Creating SQLite Database...")
print("=" * 60)

# Load processed dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

print(f"\nDataset Loaded Successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# Create SQLite database
connection = sqlite3.connect("sql/walmart.db")

print("\nConnected to SQLite Database!")

# Store data
df.to_sql(
    "walmart_sales",
    connection,
    if_exists="replace",
    index=False
)

print("\nData inserted into table: walmart_sales")

# Verify insertion
cursor = connection.cursor()
cursor.execute("SELECT COUNT(*) FROM walmart_sales")

row_count = cursor.fetchone()[0]

print(f"\nRows inside Database: {row_count}")

connection.close()

print("\nDatabase created successfully!")