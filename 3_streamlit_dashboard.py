from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "blinkit_raw_data.csv"
CLEAN = DATA_DIR / "blinkit_clean_data.csv"

st.set_page_config(page_title="Blinkit Retail Analytics", page_icon="🛒", layout="wide")

# CSS styling
st.markdown("""
<style>
/* Main dashboard background and default text */
.stApp {
    background: linear-gradient(
        135deg, #f6fff8 0%, #f4f7fb 55%, #fffaf0 100%
    );
    color: #111827;
}

.stApp p,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp span,
.stApp li {
    color: #111827;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Hero banner */
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 22px;
    color: #ffffff;
    background: linear-gradient(
        110deg, #087f5b, #16a34a 55%, #84cc16
    );
    box-shadow: 0 10px 30px rgba(5, 80, 45, .18);
    margin-bottom: 1rem;
}

/* Keep hero text white for contrast */
.hero h1,
.hero p {
    color: #ffffff !important;
}

.hero h1 {
    font-size: 2.25rem;
    margin: 0;
}

.hero p {
    font-size: 1rem;
    opacity: .95;
    margin-bottom: 0;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e6eee8;
    padding: 16px 18px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,.04);
}

/* Metric numbers, labels and deltas */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #effaf1;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #111827 !important;
}

/* Multiselect input area */
div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-color: #cbd5e1;
}

/* Selected filter tags: light green with black text */
div[data-baseweb="tag"] {
    background-color: #d1fae5 !important;
    color: #000000 !important;
}

div[data-baseweb="tag"] *,
div[data-baseweb="tag"] span {
    color: #000000 !important;
}

/* Dropdown menu and its options */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Text inputs */
.stApp input,
.stApp textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Quick Insight box */
div[data-testid="stAlert"] {
    background-color: #e0f2fe;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] strong {
    color: #111827 !important;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    color: #111827;
}

/* Selected text highlight */
::selection {
    background-color: #a7f3d0;
    color: #000000;
}

/* Small note */
.small-note {
    color: #5b6470;
    font-size: .88rem;
}
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", palette="viridis")

def ensure_clean_data():
    if not CLEAN.exists():
        if not RAW.exists():
            subprocess.run([sys.executable, str(ROOT / "1_generate_data.py")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "2_clean_data.py")], cwd=ROOT, check=True)

def load_data(use_clean):
    path = CLEAN if use_clean else RAW
    if not path.exists():
        return None
    return pd.read_csv(path)

st.markdown("""
<div class="hero">
  <h1>🛒 Blinkit Retail Analytics</h1>
  <p>Data cleaning • Exploratory analysis • Interactive visualisation • Business insights</p>
</div>
""", unsafe_allow_html=True)
st.caption("Learning project using synthetic Blinkit-style retail data — not official company data.")

with st.sidebar:
    st.header("⚙️ Data controls")
    st.write("**Step 1:** Generate synthetic raw data")
    if st.button("🎲 Generate / Reset Raw Data", use_container_width=True):
        with st.spinner("Generating dataset..."):
            subprocess.run([sys.executable, str(ROOT / "1_generate_data.py")], cwd=ROOT, check=True)
        st.success("Raw data generated.")
        st.rerun()

    st.write("**Step 2:** Clean and organise data")
    if st.button("🧹 Clean Data Now", use_container_width=True):
        with st.spinner("Cleaning data and handling missing values..."):
            if not RAW.exists():
                subprocess.run([sys.executable, str(ROOT / "1_generate_data.py")], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ROOT / "2_clean_data.py")], cwd=ROOT, check=True)
        st.success("Cleaned data saved.")
        st.rerun()

    st.divider()
    data_mode = st.radio("Dataset", ["Cleaned data", "Raw data"], index=0)
    st.caption("Tip: clean data first for more reliable charts.")

use_clean = data_mode == "Cleaned data"
df = load_data(use_clean)
if df is None:
    st.warning("No dataset found yet. Click **Generate / Reset Raw Data**, then **Clean Data Now**.")
    st.stop()

# Filter panel
st.subheader("🔎 Explore the data")
with st.expander("Filters", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    item_options = sorted(df["Item_Type"].dropna().unique().tolist())
    outlet_options = sorted(df["Outlet_Type"].dropna().unique().tolist())
    location_options = sorted(df["Outlet_Location_Type"].dropna().unique().tolist())
    fat_options = sorted(df["Item_Fat_Content"].dropna().unique().tolist())
    selected_items = c1.multiselect("Item type", item_options, default=item_options)
    selected_outlets = c2.multiselect("Outlet type", outlet_options, default=outlet_options)
    selected_locations = c3.multiselect("Location tier", location_options, default=location_options)
    selected_fat = c4.multiselect("Fat content", fat_options, default=fat_options)

filtered = df[
    df["Item_Type"].isin(selected_items)
    & df["Outlet_Type"].isin(selected_outlets)
    & df["Outlet_Location_Type"].isin(selected_locations)
    & df["Item_Fat_Content"].isin(selected_fat)
].copy()

if filtered.empty:
    st.warning("No rows match these filters. Select at least one option in each filter.")
    st.stop()

total_sales = filtered["Item_Outlet_Sales"].sum()
avg_sales = filtered["Item_Outlet_Sales"].mean()
avg_mrp = filtered["Item_MRP"].mean()
records = len(filtered)
k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 Total sales", f"₹{total_sales:,.0f}")
k2.metric("🧾 Records", f"{records:,}")
k3.metric("📊 Average sales", f"₹{avg_sales:,.2f}")
k4.metric("🏷️ Average item MRP", f"₹{avg_mrp:,.2f}")

st.divider()
left, right = st.columns([1.05, 1])
with left:
    st.subheader("📈 Choose a visualisation")
    chart_type = st.selectbox("Chart", [
        "Sales by item type", "Average sales by outlet", "Sales distribution",
        "Sales by location", "MRP vs sales", "Correlation heatmap",
        "Sales by fat content", "Sales by establishment year",
        "Item type record count", "Sales by discount", "Outlet-location heatmap",
        "Sales share by outlet", "Data preview", "Missing values"
    ])
with right:
    st.subheader("📌 Quick insight")
    top_item = filtered.groupby("Item_Type")["Item_Outlet_Sales"].sum().idxmax()
    top_outlet = filtered.groupby("Outlet_Type")["Item_Outlet_Sales"].sum().idxmax()
    st.info(f"Highest-sales item category in this selection: **{top_item}**. Highest-sales outlet type: **{top_outlet}**.")

fig, ax = plt.subplots(figsize=(11, 5.5))
if chart_type == "Sales by item type":
    s = filtered.groupby("Item_Type")["Item_Outlet_Sales"].sum().sort_values()
    s.plot(kind="barh", ax=ax)
    ax.set(title="Total Sales by Item Type", xlabel="Sales", ylabel="")
elif chart_type == "Average sales by outlet":
    sns.barplot(data=filtered, x="Outlet_Type", y="Item_Outlet_Sales", estimator="mean", errorbar=None, ax=ax)
    ax.set(title="Average Sales by Outlet Type", xlabel="", ylabel="Average sales")
    ax.tick_params(axis="x", rotation=15)
elif chart_type == "Sales distribution":
    sns.histplot(filtered["Item_Outlet_Sales"], bins=35, kde=True, ax=ax)
    ax.set(title="Sales Distribution", xlabel="Sales")
elif chart_type == "Sales by location":
    sns.boxplot(data=filtered, x="Outlet_Location_Type", y="Item_Outlet_Sales", ax=ax)
    ax.set(title="Sales by Location Tier", xlabel="", ylabel="Sales")
elif chart_type == "MRP vs sales":
    plot_df = filtered.sample(min(4000, len(filtered)), random_state=42)
    sns.scatterplot(data=plot_df, x="Item_MRP", y="Item_Outlet_Sales", hue="Outlet_Type", alpha=.65, ax=ax)
    ax.set(title="Item MRP vs Sales")
elif chart_type == "Correlation heatmap":
    corr = filtered.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlations")
elif chart_type == "Sales by fat content":
    sns.barplot(data=filtered, x="Item_Fat_Content", y="Item_Outlet_Sales", estimator="mean", errorbar=None, ax=ax)
    ax.set(title="Average Sales by Fat Content", xlabel="", ylabel="Average sales")
elif chart_type == "Sales by establishment year":
    s = filtered.groupby("Outlet_Establishment_Year")["Item_Outlet_Sales"].mean().sort_index()
    s.plot(marker="o", ax=ax)
    ax.set(title="Average Sales by Establishment Year", xlabel="Year", ylabel="Average sales")
elif chart_type == "Item type record count":
    filtered["Item_Type"].value_counts().sort_values().plot(kind="barh", ax=ax)
    ax.set(title="Records by Item Type", xlabel="Record count", ylabel="")
elif chart_type == "Sales by discount":
    if "Discount_Percent" not in filtered.columns:
        filtered["Discount_Percent"] = filtered["Discount_Rate"] * 100
    sns.boxplot(data=filtered, x="Discount_Percent", y="Item_Outlet_Sales", ax=ax)
    ax.set(title="Sales by Discount Level", xlabel="Discount (%)", ylabel="Sales")
elif chart_type == "Outlet-location heatmap":
    pivot = filtered.pivot_table(index="Outlet_Type", columns="Outlet_Location_Type", values="Item_Outlet_Sales", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Average Sales: Outlet Type vs Location")
elif chart_type == "Sales share by outlet":
    s = filtered.groupby("Outlet_Type")["Item_Outlet_Sales"].sum().sort_values(ascending=False)
    s.plot(kind="pie", autopct="%1.1f%%", startangle=90, ylabel="", ax=ax)
    ax.set_title("Sales Share by Outlet")
elif chart_type == "Data preview":
    plt.close(fig)
    st.dataframe(filtered.head(1000), use_container_width=True, height=360)
    st.download_button("⬇️ Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "blinkit_filtered_data.csv", "text/csv")
    st.stop()
elif chart_type == "Missing values":
    missing = filtered.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        ax.text(.5, .5, "No missing values in this selection", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Missing cell count")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("📋 Data quality and summary"):
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")
    q2.metric("Duplicate rows", f"{int(filtered.duplicated().sum()):,}")
    q3.metric("Columns", str(filtered.shape[1]))
    st.dataframe(filtered.describe(include="all").transpose(), use_container_width=True)

st.download_button(
    "⬇️ Download current filtered data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="blinkit_filtered_data.csv",
    mime="text/csv"
)