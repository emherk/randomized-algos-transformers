# read in all fil names in a directory 
import os
import glob
import re
import numpy as np
import pandas as pd

# Directory containing the checkpoint files
DIR = 'checkpoints/*'

# Get all files in the directory except files that have 'log_dict' in their name
files = [f for f in glob.glob(DIR) if 'log_dict' not in f]
print(files)

df = pd.DataFrame(files, columns=['file_path'])
df[['path', 'recall', 'run_id', 'probabilistic','q', 'c']]= df['file_path'].str.replace('.npy', '').str.split(r'_(?!seed)|(?<!single)_', expand=True)
df = df.drop(columns=['path', 'recall', 'file_path'])

df.sort_values(by=['probabilistic', 'q', 'c'], inplace=True)

print(df)

