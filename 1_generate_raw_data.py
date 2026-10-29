from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 50_000
OUT = Path("data")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

item_types = [
    "Fruits and Vegetables", "Snack Foods", "Household", "Dairy", "Canned",
    "Baking Goods", "Frozen Foods", "Meat", "Soft Drinks", "Health and Hygiene",
    "Hard Drinks", "Breads", "Others", "Breakfast", "Seafood"
]
fat_contents = ["Low Fat", "Regular"]
outlet_types = ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"]
outlet_sizes = ["Small", "Medium", "High"]
locations = ["Tier 1", "Tier 2", "Tier 3"]
est_years = np.arange(1985, 2023)

item_type = rng.choice(item_types, N_ROWS)
outlet_type = rng.choice(outlet_types, N_ROWS, p=[.45, .22, .18, .15])
item_mrp = np.round(rng.uniform(25, 270, N_ROWS), 2)
item_visibility = np.round(rng.beta(2, 8, N_ROWS), 4)
item_weight = np.round(rng.normal(12.5, 4.5, N_ROWS).clip(2, 30), 2)
fat = rng.choice(fat_contents, N_ROWS, p=[.62, .38])
outlet_size = rng.choice(outlet_sizes, N_ROWS, p=[.28, .5, .22])
tier = rng.choice(locations, N_ROWS, p=[.32, .38, .30])
outlet_year = rng.choice(est_years, N_ROWS)
discount = rng.choice([0, .05, .10, .15, .20, .25], N_ROWS, p=[.32, .15, .2, .15, .12, .06])
sales = item_mrp * rng.uniform(.65, 1.45, N_ROWS) * (1 - discount)
sales *= np.where(outlet_type == "Supermarket Type3", 1.12, 1.0)
sales *= np.where(tier == "Tier 1", 1.08, np.where(tier == "Tier 3", .94, 1.0))
sales = np.round(sales.clip(10, 800), 2)

df = pd.DataFrame({
    "Item_Identifier": [f"FD{n:06d}" for n in rng.integers(1, 999999, N_ROWS)],
    "Item_Type": item_type,
    "Item_Fat_Content": fat,
    "Item_Weight": item_weight,
    "Item_Visibility": item_visibility,
    "Item_MRP": item_mrp,
    "Discount_Rate": discount,
    "Outlet_Identifier": [f"OUT{n:03d}" for n in rng.integers(1, 101, N_ROWS)],
    "Outlet_Establishment_Year": outlet_year,
    "Outlet_Size": outlet_size,
    "Outlet_Location_Type": tier,
    "Outlet_Type": outlet_type,
    "Item_Outlet_Sales": sales
})

# Intentionally add realistic data-quality problems for the cleaning exercise.
for col, frac in {"Item_Weight": .055, "Outlet_Size": .045, "Item_Visibility": .02}.items():
    idx = rng.choice(df.index, size=int(N_ROWS * frac), replace=False)
    df.loc[idx, col] = np.nan

# Add inconsistent category labels and duplicate rows.
fat_idx = rng.choice(df.index, size=800, replace=False)
df.loc[fat_idx[:270], "Item_Fat_Content"] = "LF"
df.loc[fat_idx[270:540], "Item_Fat_Content"] = "low fat"
df.loc[fat_idx[540:], "Item_Fat_Content"] = "reg"
df = pd.concat([df, df.sample(250, random_state=SEED)], ignore_index=True)

path = OUT / "blinkit_raw_data.csv"
df.to_csv(path, index=False)
print(f"Created {path.resolve()} with {len(df):,} rows and {len(df.columns)} columns.")
print("Note: data is synthetic; it is not official Blinkit data.")