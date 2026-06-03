import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import random
import time

# Import the Dataset class and Transforms we built yesterday!
from dataset import SkinLesionDataset, train_transforms, val_test_transforms, LABEL_COL, NUM_CLASSES

# Reproducibility & Device Setup
# Set "seeds" so the random shuffling happens the exact same way every time we run this
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Automatically use the graphics card (GPU) if one is available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training Hyperparameters 
BATCH_SIZE = 32
LEARNING_RATE_PHASE_1 = 0.001
LEARNING_RATE_PHASE_2 = 0.0001
EPOCHS_PHASE_1 = 5
EPOCHS_PHASE_2 = 3

# --- 3. Load the Datasets & Fix Imbalance ---

# Create the full training dataset
full_train_dataset = SkinLesionDataset(
    csv_file='SkinLesions_train.csv', 
    img_folder='Training_dataset_SkinLesions', 
    transform=train_transforms
)

# How to fix the 53.9x Class Imbalance:
# 1. Count how many times each label appears
class_counts = np.bincount(full_train_dataset.labels)

# 2. Calculate a "weight" for each class. (Rare classes get huge weights, common ones get tiny weights)
class_weights = 1.0 / class_counts

# 3. Assign a specific weight to every single image in the dataset based on its label
sample_weights = [class_weights[label] for label in full_train_dataset.labels]

# 4. Create the rigged conveyor belt (The Sampler)
sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)

# 5. Build the final DataLoader using our rigged sampler
train_loader = DataLoader(
    full_train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler # We use the sampler instead of shuffle=True
)

print(f"\nSuccessfully loaded {len(full_train_dataset)} images into the DataLoader.")
print("Class Imbalance has been neutralized via WeightedRandomSampler.")

# --- 4. Build the Model (Transfer Learning) ---
print("\nLoading pretrained ResNet18...")

# 1. Load the expert model pre-trained on millions of images
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze the early layers (the "eyes") so they don't forget how to see basic shapes
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the very last layer (the "mouth") to output our 8 skin lesion classes
# ResNet18's final layer receives 512 signals, so we map those 512 signals to our NUM_CLASSES (8)
model.fc = nn.Linear(512, NUM_CLASSES)

# 4. Send the model to the graphics card (if you have one) or keep it on the CPU
model = model.to(device)


# --- 5. Define the Grader and the Optimizer ---
# The Grader (Loss Function): Calculates exactly how wrong the model's guess was
criterion = nn.CrossEntropyLoss()

# The Student (Optimizer): Tweaks the math in our brand new final layer to get a better score
# We give it our Phase 1 learning speed (0.001)
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_PHASE_1)

print("Model built and ready for training!")