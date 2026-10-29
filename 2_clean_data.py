from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
source = DATA_DIR / "blinkit_raw_data.csv"
target = DATA_DIR / "blinkit_clean_data.csv"

if not source.exists():
    raise FileNotFoundError("Raw data not found. First run: python 1_generate_data.py")

df = pd.read_csv(source)
rows_before = len(df)
missing_before = int(df.isna().sum().sum())
duplicates_before = int(df.duplicated().sum())

# Standardise column names and text categories.
df.columns = [c.strip() for c in df.columns]
df["Item_Fat_Content"] = (
    df["Item_Fat_Content"].astype("string").str.strip().str.lower()
    .replace({"lf": "low fat", "low fat": "low fat", "reg": "regular"})
    .str.title()
)
df["Item_Type"] = df["Item_Type"].astype("string").str.strip().str.title()
for col in ["Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]:
    df[col] = df[col].astype("string").str.strip()

# Remove exact duplicate rows.
df = df.drop_duplicates().copy()

# Convert numeric columns safely.
numeric_cols = [
    "Item_Weight", "Item_Visibility", "Item_MRP", "Discount_Rate",
    "Outlet_Establishment_Year", "Item_Outlet_Sales"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Impute missing values with group median, then overall median.
df["Item_Weight"] = df.groupby("Item_Type")["Item_Weight"].transform(
    lambda s: s.fillna(s.median())
)
df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].median())
df["Item_Visibility"] = df["Item_Visibility"].fillna(df["Item_Visibility"].median())
df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Size"].mode().iloc[0])

# Add useful derived fields.
df["Outlet_Age"] = (2026 - df["Outlet_Establishment_Year"]).clip(lower=0)
df["Sales_Per_MRP"] = (df["Item_Outlet_Sales"] / df["Item_MRP"]).round(3)
df["Discount_Percent"] = (df["Discount_Rate"] * 100).round(1)

# Keep rows with essential business measures and valid ranges.
df = df.dropna(subset=["Item_Identifier", "Item_Type", "Item_MRP", "Item_Outlet_Sales"])
df = df[df["Item_MRP"].between(0, 1000)]
df = df[df["Item_Outlet_Sales"].between(0, 2000)]
df = df.sort_values(["Outlet_Type", "Item_Type", "Item_Outlet_Sales"], ascending=[True, True, False])
df = df.reset_index(drop=True)

# Organise columns for easier analysis.
ordered = [
    "Item_Identifier", "Item_Type", "Item_Fat_Content", "Item_Weight",
    "Item_Visibility", "Item_MRP", "Discount_Rate", "Discount_Percent",
    "Outlet_Identifier", "Outlet_Establishment_Year", "Outlet_Age",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
    "Item_Outlet_Sales", "Sales_Per_MRP"
]
df = df[[c for c in ordered if c in df.columns]]

df.to_csv(target, index=False)
report = f"""BLINKIT-STYLE DATA CLEANING REPORT (SYNTHETIC DATA)
====================================================
Input file: {source}
Output file: {target}
Rows before cleaning: {rows_before:,}
Rows after cleaning:  {len(df):,}
Duplicate rows found before cleaning: {duplicates_before:,}
Missing cells before cleaning: {missing_before:,}
Missing cells after cleaning: {int(df.isna().sum().sum()):,}
Columns in cleaned dataset: {len(df.columns)}
Output missing values by column:
{df.isna().sum().to_string()}
"""
(REPORT_DIR / "cleaning_report.txt").write_text(report, encoding="utf-8")
print(report)
print(f"Saved cleaned data to {target.resolve()}")