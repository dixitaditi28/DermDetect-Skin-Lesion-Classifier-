import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import random

from dataset import (
    SkinLesionDataset,
    SkinLesionFolderDataset,
    train_transforms,
    val_test_transforms,
    NUM_CLASSES
)

# ── Reproducibility & Device Setup ────────────────────────────────────────────
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE           = 32
LEARNING_RATE_PHASE_1 = 0.001
LEARNING_RATE_PHASE_2 = 0.0001
EPOCHS_PHASE_1       = 5
EPOCHS_PHASE_2       = 3

# ── 1. Load Training Dataset & Fix Class Imbalance ───────────────────────────
full_train_dataset = SkinLesionDataset(
    csv_file='SkinLesions_train.csv',
    img_folder='Training_dataset_SkinLesions',
    transform=train_transforms
)

# Rare classes get high weights so the sampler picks them more often
class_counts  = np.bincount(full_train_dataset.labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in full_train_dataset.labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    full_train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler       # sampler handles shuffling — don't pass shuffle=True
)

print(f"\nSuccessfully loaded {len(full_train_dataset)} images into the DataLoader.")
print("Class Imbalance has been neutralized via WeightedRandomSampler.")

# ── 2. Load Validation Dataset (Folder-based — no CSV needed) ────────────────
# FIX: SkinLesionFolderDataset reads files directly from the folder,
# so the train/test filename prefix mismatch is no longer a problem.
print("\nSetting up the Validation Exam...")

val_dataset = SkinLesionFolderDataset(
    img_folder='Test_dataset_SkinLesions',
    transform=val_test_transforms   # strict rules — no random flipping
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False                   # no shuffling needed for evaluation
)

# ── 3. Build the Model (Transfer Learning with ResNet18) ─────────────────────
print("\nLoading pretrained ResNet18...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all early layers so they keep their general vision knowledge
for param in model.parameters():
    param.requires_grad = False

# Replace the final classification head for our 8 skin lesion classes
model.fc = nn.Linear(512, NUM_CLASSES)
model = model.to(device)

# ── 4. Loss Function & Optimizer ──────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# Phase 1: only train the new final layer (everything else is frozen)
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_PHASE_1)

print("Model built and ready for training!")

# ── 5. Training Loop — Phase 1 (Frozen Backbone) ────────────────────────────
print("\n--- Starting Phase 1 Training (Frozen Eyes) ---")
best_val_acc = 0.0

for epoch in range(EPOCHS_PHASE_1):

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    running_loss  = 0.0
    correct_train = 0
    total_train   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # 1. Erase scratchpad
        outputs = model(images)            # 2. Make predictions
        loss = criterion(outputs, labels)  # 3. Score the predictions
        loss.backward()                    # 4. Calculate gradients
        optimizer.step()                   # 5. Update the final layer only

        running_loss  += loss.item() * images.size(0)
        _, predicted   = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train   += labels.size(0)

    train_loss = running_loss / total_train
    train_acc  = (correct_train / total_train) * 100

    # ── Validation ────────────────────────────────────────────────────────────
    # NOTE: val labels are -1 (no ground truth), so we track loss only.
    # Swap in a labelled test CSV later to get real accuracy here.
    model.eval()
    running_val_loss = 0.0
    total_val        = 0

    with torch.no_grad():                  # no gradients needed for evaluation
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)

            running_val_loss += outputs.size(0)  # placeholder count
            total_val        += images.size(0)

    print(
        f"Epoch [{epoch+1}/{EPOCHS_PHASE_1}] | "
        f"Train Loss: {train_loss:.3f} | "
        f"Train Acc: {train_acc:.1f}% | "
        f"Val Batches Processed: {total_val} images"
    )

# ── 6. Phase 2 — Unfreeze & Fine-tune the Full Network ───────────────────────
print("\n--- Starting Phase 2 Training (Full Fine-tuning) ---")

# Unfreeze ALL layers so the whole network can adapt to skin lesion images
for param in model.parameters():
    param.requires_grad = True

# Use a much smaller learning rate to avoid overwriting the pretrained knowledge
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE_2)

for epoch in range(EPOCHS_PHASE_2):

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    running_loss  = 0.0
    correct_train = 0
    total_train   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * images.size(0)
        _, predicted   = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train   += labels.size(0)

    train_loss = running_loss / total_train
    train_acc  = (correct_train / total_train) * 100

    print(
        f"Epoch [{epoch+1}/{EPOCHS_PHASE_2}] | "
        f"Train Loss: {train_loss:.3f} | "
        f"Train Acc: {train_acc:.1f}%"
    )

    # --- Phase 2 Validation Exam ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
            
    val_acc = (correct_val / total_val) * 100
    val_loss = running_val_loss / total_val
    print(f"   ↳ Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.1f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_skin_model.pth')
        print("   🌟 New high score! Saved best_skin_model.pth")

# (This goes at the very absolute bottom, un-indented, completely outside the loop)
print("\n🎉 Training Complete! Your best model is saved as 'best_skin_model.pth'")

# ── 7. Save the Trained Model ─────────────────────────────────────────────────
torch.save(model.state_dict(), 'skin_lesion_resnet18.pth')
print("\nTraining complete. Model saved to skin_lesion_resnet18.pth")




# --- 8. Phase 2: Unfreeze the Eyes (Fine-tuning) ---
print("\n--- Starting Phase 2: Fine-Tuning Entire Model ---")

# Unfreeze all layers so the model can learn specific skin textures
for param in model.parameters():
    param.requires_grad = True

# Create a new Optimizer with a tiny learning rate (Phase 2 speed)
optimizer_phase2 = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE_2)

# Loop 3 times (Phase 2)
for epoch in range(EPOCHS_PHASE_2):
    
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer_phase2.zero_grad()             
        outputs = model(images)           
        loss = criterion(outputs, labels) 
        loss.backward()                   
        optimizer_phase2.step()                  

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) 
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = (correct_train / total_train) * 100
    train_loss = running_loss / total_train
    
    print(f"Phase 2 - Epoch [{epoch+1}/{EPOCHS_PHASE_2}] | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.1f}%")

    # --- Phase 2 Validation Exam ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
            
    val_acc = (correct_val / total_val) * 100
    val_loss = running_val_loss / total_val
    print(f"   ↳ Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.1f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_skin_model.pth')
        print("   🌟 New high score! Saved best_skin_model.pth")

print("\n🎉 Training Complete! Your best model is saved as 'best_skin_model.pth'")