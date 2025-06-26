# read in all fil names in a directory
import os
import glob
import re
import numpy as np
import pandas as pd

# Directory containing the checkpoint files
DIR = "checkpoints/*"

# Get all files in the directory except files that have 'log_dict' in their name
files = [f for f in glob.glob(DIR) if "log_dict" not in f]
print(files)

df = pd.DataFrame(files, columns=["file_path"])
df[["path", "recall", "run_id", "probabilistic", "q", "c"]] = (
    df["file_path"]
    .str.replace(".npy", "")
    .str.split(r"_(?!seed)|(?<!single)_", expand=True)
)
df = df.drop(columns=["path", "recall"])
df['q'] = df['q'].str.replace("q", "").astype(int)
df['c'] = df['c'].str.replace("c", "").astype(int)

df.sort_values(by=["probabilistic", "q", "c"], inplace=True)

print(df)

df_duplicates = df[df.duplicated(subset=["probabilistic", "q", "c"])]
# move duplicates to a separate folder
duplicates_dir = "checkpoints_duplicate"
os.makedirs(duplicates_dir, exist_ok=True)
for _, row in df_duplicates.iterrows():
    file_path = row['file_path']
    new_file_path = os.path.join(duplicates_dir, os.path.basename(file_path))
    os.rename(file_path, new_file_path)

df = df.drop_duplicates(subset=["probabilistic", "q", "c"])
df.to_csv("organized_runs.csv", index=False)

# i want to have all the values in the dataframe
probabilistic = ["single_seed", "deterministic", "random"]
num_token = [8, 10, 12, 14, 16, 18, 20]
p = [1, 16, 32, 100]
# get which combinations are missing

missing_combinations = []
for prob in probabilistic:
    for context_length in num_token:
        for q_val in p:
            if not ((df["probabilistic"] == prob) & (df["q"] == q_val) & (df["c"] == context_length)).any():
                missing_combinations.append((prob,  q_val, context_length))
df_missing = pd.DataFrame(missing_combinations, columns=["probabilistic", "q", "c"])
df_missing.sort_values(by=["probabilistic", "q", "c"], inplace=True)

print("Missing combinations:")
print(df_missing)

df_missing.to_csv("df_missing.csv", index=False)
