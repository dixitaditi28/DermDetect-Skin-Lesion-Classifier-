import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. Setup & Configuration ---
# Make sure this perfectly matches the file your training script saved!
MODEL_PATH = 'skin_lesion_resnet18.pth'
NUM_CLASSES = 8
device = torch.device("cpu")

# We use the strict exam filters (no random flipping for the final test!)
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. Load Your Trained Brain ---
print(f"Loading your trained model from {MODEL_PATH}...")
model = models.resnet18(weights=None) # We don't need the generic expert anymore
model.fc = nn.Linear(512, NUM_CLASSES) # Rebuild our custom 8-disease mouth
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Put it in strict exam mode

# --- 3. Pick a Patient Image to Diagnose ---
# Let's grab the very first image from your test folder
IMAGE_PATH = 'Test_dataset_SkinLesions/SkinLesions_test_0001.png'

if not os.path.exists(IMAGE_PATH):
    print(f"\nERROR: Could not find the image at {IMAGE_PATH}")
    print("Please check the exact filename in your Test folder and update IMAGE_PATH!")
    exit()

print(f"Diagnosing image: {IMAGE_PATH}...")
image = Image.open(IMAGE_PATH).convert('RGB')
image_tensor = val_test_transforms(image).unsqueeze(0) # Add a fake "batch" dimension of 1

# --- 4. Make the Prediction ---
with torch.no_grad():
    outputs = model(image_tensor)
    
    # Calculate the confidence percentages using Softmax math
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0) * 100
    
    # Find the highest score
    confidence, predicted_idx = torch.max(probabilities, 0)
    
    # Remember we subtracted 1 during training? We add 1 back to match your CSV Target numbers
    predicted_target = predicted_idx.item() + 1 

print("\n" + "="*40)
print("🩺 DIAGNOSIS RESULTS")
print("="*40)
print(f"Predicted Target Class : {predicted_target}")
print(f"Model Confidence       : {confidence.item():.2f}%")
print("-" * 40)

# Optional: Print the runner-up guesses
print("Top 3 Guesses:")
top3_prob, top3_idx = torch.topk(probabilities, 3)
for i in range(3):
    target_num = top3_idx[i].item() + 1
    prob = top3_prob[i].item()
    print(f"  Target {target_num}: {prob:.2f}%")
print("="*40 + "\n")