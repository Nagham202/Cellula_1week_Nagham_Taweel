import pandas as pd

df = pd.read_csv("toxic_data.csv")

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 200)

print("Shape (rows, columns):", df.shape)

print("\nColumn types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nFully duplicated rows:", df.duplicated().sum())

print("\nUnique values per column:")
print(df.nunique())

for col in df.columns:
    if df[col].nunique() <= 20:
        print(f"\n--- Value counts for '{col}' ---")
        counts = df[col].value_counts()
        percents = (df[col].value_counts(normalize=True) * 100).round(1)
        print(pd.DataFrame({"count": counts, "percent": percents}))
