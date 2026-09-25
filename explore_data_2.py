import pandas as pd

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("toxic_data.csv")

unique_df = df.drop_duplicates()
print("Rows before:", len(df), "| after removing duplicates:", len(unique_df))
comparison = pd.DataFrame({
    "before": df["Toxic Category"].value_counts(),
    "after": unique_df["Toxic Category"].value_counts(),
})
print(comparison)

small_classes = ["Elections", "Sex-Related Crimes", "Child Sexual Exploitation", "Suicide & Self-Harm"]
for category in small_classes:
    print(f"\n=== {category} ===")
    rows = unique_df[unique_df["Toxic Category"] == category]
    print(rows[["query", "image descriptions"]].to_string())

table = pd.crosstab(unique_df["image descriptions"], unique_df["Toxic Category"])
table.columns = [name[:10] for name in table.columns]
print("\n", table)

lengths = unique_df["query"].str.split().str.len()
print("\nQuery length (words):")
print(lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
