import os
import pandas as pd
import matplotlib.pyplot as plt

# we load the csv  File. note that its saved as csv and not as an xlxs because its microsoft proprietary format and we want to avoid any issues with that. 
# also, csv is a more common format for data science projects. 
# if you have an xlsx file, you can easily convert it to csv using excel or google sheets.
csv_path = 'SkinLesions_train.csv'

try:
    # Read the CSV file into a pandas "DataFrame"
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded! The table has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nHere are the first 3 rows:")
    print(df.head(3))
    
except FileNotFoundError:
    print(f"\nERROR: I could not find '{csv_path}'.")
    exit()

# --- Step 3: Find the important columns ---
possible_label_cols = ['label', 'class', 'dx', 'diagnosis', 'category', 'target']
possible_file_cols = ['image_id', 'filename', 'image', 'path', 'id']

LABEL_COL = None
FILE_COL = None

for col in df.columns:
    clean_col_name = col.lower()
    
    if clean_col_name in possible_label_cols:
        LABEL_COL = col
        
    if clean_col_name in possible_file_cols:
        FILE_COL = col

print(f"\nSuccess! Detected Label Column: {LABEL_COL}")
print(f"Success! Detected File Column: {FILE_COL}")


