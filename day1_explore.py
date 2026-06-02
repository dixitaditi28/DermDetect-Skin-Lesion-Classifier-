import os 
import pandas as pd #takes spreadsheet from hard drive and loads into memory as a grid called dataframe df in short
import matplotlib.pyplot as plt

# we load the csv  File. note that its saved as csv and not as an xlxs because its microsoft proprietary format and we want to avoid any issues with that. 
# also, csv is a more common format for data science projects. 
# if you have an xlsx file, you can easily convert it to csv using excel or google sheets.
csv_path = 'SkinLesions_train.csv'


# we wrap the code in a try/except block this acts as a safety net
# basically means try to open the file, but if it doesn't exist, catch the error and print a friendly message instead of crashing ugly red text onto the screen.
try:
    # Read the CSV file into a pandas df
    df = pd.read_csv(csv_path)
    
    # df tells us no. of rows and cols. .shape is a special attribute of dataframes that gives us this info as a tuple (rows, cols).
    print(f"Dataset loaded! The table has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nHere are the first 3 rows:")
    print(df.head(3)) #head peeks at the top 3 rows of the spreadsheet
    
except FileNotFoundError:
    print(f"\nERROR: I could not find '{csv_path}'.")
    print ("Please make sure the file is in the exact same folder")
    exit() 
    # stops the script here if we cant find the data

# NOW WE AUTO DETECT THE IMP COLS
# Find the important columns 
possible_label_cols = ['label', 'class', 'dx', 'diagnosis', 'category', 'target']
possible_file_cols = ['image_id', 'filename', 'image', 'path', 'id']

# we store the correct column names in these variables
LABEL_COL = None
FILE_COL = None


for col in df.columns:
    clean_col_name = col.lower() #convert column name into lowercase in case its capitalized.
    
    
    if clean_col_name in possible_label_cols:
        LABEL_COL = col
        
    if clean_col_name in possible_file_cols:
        FILE_COL = col

print(f"\nSuccess! Detected Label Column: {LABEL_COL}")
print(f"Success! Detected File Column: {FILE_COL}")

# Now that we know where the data is, we count the classes and draw a graph 

# CHECK FOR CLASS IMBALANCE = which is in 20k dataset mei if 15k is A and 500 is B then theres a imbalance
# df[LABEL_COL] grabs just the 'Target' column from our spreadsheet.
# .value_counts() automatically groups identical numbers together and counts them.
class_counts = df[LABEL_COL].value_counts()

print("\n--- Class Distribution ---")
print(class_counts)

# we check for severe imbalance (ratio > 3x)
max_class = class_counts.max()
min_class = class_counts.min()
if max_class / min_class > 3:
    print(f"\nWARNING: Severe class imbalance detected! (Ratio: {max_class/min_class:.1f}x)")
    print("We will fix this on Day 3 using a Weighted Random Sampler.")

#next we Set up the canvas
# plt is our nickname for Matplotlib. 
# figsize=(10, 6) tells it to create a rectangular canvas 10 inches wide, 6 inches tall.
plt.figure(figsize=(10, 6))

# Draw the chart
# kind='bar' tells pandas to draw vertical bars using the counts we just calculated.
class_counts.plot(kind='bar')

# Add labels so humans can read it
plt.title('Distribution of Skin Lesions in Dataset')
plt.xlabel('Lesion Type (Target Code)')
plt.ylabel('Number of Images')

# Clean up and save
# tight_layout() makes sure no text gets cut off the edges of the image.
plt.tight_layout()
# savefig() actually saves the drawing to your hard drive as a PNG file.
plt.savefig('class_distribution.png')

print("\nSuccess! Saved a bar chart to 'class_distribution.png' in your folder.")


