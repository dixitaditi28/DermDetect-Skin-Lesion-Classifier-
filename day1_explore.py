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

# --- Step 4: Count the classes and draw a graph ---

# 1. Count the M&Ms
# df[LABEL_COL] grabs just the 'Target' column from our spreadsheet.
# .value_counts() automatically groups identical numbers together and counts them.
class_counts = df[LABEL_COL].value_counts()

print("\n--- Class Distribution ---")
print(class_counts)

# Check for severe imbalance (ratio > 3x)
max_class = class_counts.max()
min_class = class_counts.min()
if max_class / min_class > 3:
    print(f"\nWARNING: Severe class imbalance detected! (Ratio: {max_class/min_class:.1f}x)")
    print("We will fix this on Day 3 using a Weighted Random Sampler.")

# 2. Set up the canvas
# plt is our nickname for Matplotlib. 
# figsize=(10, 6) tells it to create a rectangular canvas 10 inches wide, 6 inches tall.
plt.figure(figsize=(10, 6))

# 3. Draw the chart
# kind='bar' tells pandas to draw vertical bars using the counts we just calculated.
class_counts.plot(kind='bar')

# 4. Add labels so humans can read it
plt.title('Distribution of Skin Lesions in Dataset')
plt.xlabel('Lesion Type (Target Code)')
plt.ylabel('Number of Images')

# 5. Clean up and save
# tight_layout() makes sure no text gets cut off the edges of the image.
plt.tight_layout()
# savefig() actually saves the drawing to your hard drive as a PNG file.
plt.savefig('class_distribution.png')

print("\nSuccess! Saved a bar chart to 'class_distribution.png' in your folder.")


