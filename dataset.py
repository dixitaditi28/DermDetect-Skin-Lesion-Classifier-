import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_COL   = 'Target'
FILE_COL    = 'ID'
NUM_CLASSES = 8

# ── Transforms ─────────────────────────────────────────────────────────────────
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Training: includes augmentation to prevent memorization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Validation/Test: strict formatting, no random flipping
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ── Dataset: CSV-based (Training) ──────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    """Loads images by matching IDs from a CSV file to files on disk."""

    def __init__(self, csv_file, img_folder, transform=None):
        self.metadata    = pd.read_csv(csv_file)
        self.img_folder  = img_folder
        self.transform   = transform
        self.image_paths = []
        self.labels      = []

        print(f"Scanning folder: {img_folder}...")

        for _, row in self.metadata.iterrows():
            extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
            found_path = None

            for ext in extensions:
                temp_path = os.path.join(img_folder, str(row[FILE_COL]) + ext)
                if os.path.exists(temp_path):
                    found_path = temp_path
                    break

            if found_path:
                self.image_paths.append(found_path)
                self.labels.append(int(row[LABEL_COL]) - 1)  # 0-indexed

        total = len(self.metadata)
        valid = len(self.image_paths)
        print(f"Found {valid} valid images out of {total} rows.")
        if valid < total:
            print(f"Skipped {total - valid} missing images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label    = self.labels[index]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            print(f"\nWarning: Corrupt image at {img_path}. Falling back to index 0.")
            return self.__getitem__(0)

        if self.transform:
            image = self.transform(image)
        return image, label


# ── Dataset: Folder-based (Validation / Test) ──────────────────────────────────
class SkinLesionFolderDataset(Dataset):
    """Loads images directly from a folder — no CSV or labels needed."""

    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform  = transform

        self.image_files = sorted([
            f for f in os.listdir(img_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Found {len(self.image_files)} images in '{img_folder}'")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.image_files[idx])
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, -1  # -1 = dummy label, no ground truth for test set


# ── Smoke Test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_dataset = SkinLesionDataset(
        csv_file='SkinLesions_train.csv',
        img_folder='Training_dataset_SkinLesions',
        transform=train_transforms
    )

    first_image, first_label = train_dataset[0]
    print("\n--- Test Results ---")
    print(f"Image tensor shape : {first_image.shape}")
    print(f"Label (0-indexed)  : {first_label}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch_images, batch_labels = next(iter(train_loader))

    print("\n--- DataLoader Test ---")
    print(f"Batch image shape : {batch_images.shape}")  # expect [32, 3, 224, 224]
    print(f"Batch labels      : {batch_labels}")