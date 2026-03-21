import pandas as pd

input_file = "/home/vu/lfd/data/preprocessed/both_preprocessed_normalized.csv"   # <-- change this to your file name
chunksize = 100000

# ---- Save first 100 rows ----
first_chunk = pd.read_csv(input_file, nrows=100)
first_chunk.to_csv("first_100.csv", index=False)

# ---- Save last 100 rows ----
last_rows = []

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    last_rows.append(chunk.tail(100))

# Combine last partials & get final 100 rows
last_100 = pd.concat(last_rows).tail(100)
last_100.to_csv("last_100.csv", index=False)

print("Saved first_100.csv and last_100.csv successfully!")
