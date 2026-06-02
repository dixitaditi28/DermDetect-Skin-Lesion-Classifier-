import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

#Constants from Day 1
LABEL_COL = 'Target'
FILE_COL = 'ID'
NUM_CLASSES = 8

# --- Image Transforms (The Photoshop Filters) ---
# ImageNet standard colors for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training Transforms: Includes "Data Augmentation" to prevent memorization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)
])

# Validation/Test Transforms: Strict formatting, NO random flipping!
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- Step 2: The Dataset Class (The Librarian) ---
class SkinLesionDataset(Dataset):
    
    # 1. The Setup
    def __init__(self, csv_file, img_folder, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        
        # We will store the exact file path and label for every valid image here
        self.image_paths = []
        self.labels = []
        
        print(f"Scanning folder: {img_folder}...")
        
        # Loop through every row in the CSV
        for index, row in self.metadata.iterrows():
            # Add '.png' to the ID (e.g., 'SkinLesions_train_0001' becomes 'SkinLesions_train_0001.png')
            img_name = str(row[FILE_COL]) + '.png' 
            img_path = os.path.join(self.img_folder, img_name)
            
            # Check if the file actually exists on your hard drive
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                
                # Subtract 1 so labels start at 0 instead of 1
                label = int(row[LABEL_COL]) - 1 
                self.labels.append(label)
        
        total_rows = len(self.metadata)
        valid_images = len(self.image_paths)
        print(f"Found {valid_images} valid images out of {total_rows} rows.")
        if valid_images < total_rows:
            print(f"Skipped {total_rows - valid_images} missing images.")

    # 2. The Counter
    def __len__(self):
        return len(self.image_paths)

    # 3. The Fetcher
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        try:
            # Open the image file using PIL (the digital darkroom)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If the image is corrupt, print a warning and just grab the very first image instead to prevent a crash
            print(f"\nWarning: Corrupt image {img_path}. Skipping.")
            return self.__getitem__(0)
            
        # Apply our Photoshop filters (Transforms)
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
# --- Step 3: Test the Librarian ---
# This block only runs if we execute this specific file
if __name__ == '__main__':
    
    # 1. Hire the librarian for our training dataset
    train_dataset = SkinLesionDataset(
        csv_file='SkinLesions_train.csv', 
        img_folder='Training_dataset_SkinLesions', 
        transform=train_transforms
    )
    
    # 2. Ask the librarian for the very first image (Index 0)
    first_image, first_label = train_dataset[0]
    
    # 3. Print the results
    print("\n--- Test Results ---")
    print(f"Success! Image converted to Tensor math. Shape: {first_image.shape}")
    print(f"Label integer (Should be exactly 1 less than the CSV target): {first_label}")    
    # 4. Build the Conveyor Belt (DataLoader)
    # batch_size=32 means hand the network 32 images at a time
    # shuffle=True acts like a dealer shuffling a deck of cards
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 5. Turn on the conveyor belt and grab the very first tray
    # iter() turns the belt on, next() grabs the first item that comes off
    batch_images, batch_labels = next(iter(train_loader))
    
    print("\n--- Conveyor Belt Test ---")
    # We expect [32 images, 3 colors, 224 width, 224 height]
    print(f"Batch image shape: {batch_images.shape}")
    print(f"Batch labels:\n{batch_labels}")