# 🔬 lesion-classifier
 
> A skin lesion image classifier built with PyTorch, ResNet18, and Flask.
> Upload a photo — get an AI prediction with confidence scores.
 
 
---
 
## What this project does
 
lesion-classifier is a web application that takes a photo of a skin lesion and predicts
what type it is — melanoma, nevus, basal cell carcinoma, and more — along with a
confidence percentage for each prediction.
 
You upload an image in the browser. The Flask backend passes it through a fine-tuned
ResNet18 neural network. The top 3 predictions come back as a visual bar chart in
under a second.
 
> ⚠️ **Medical disclaimer:** lesion-classifier is a learning project only. It is not
> clinically validated and must never be used for real medical decisions. Always
> consult a qualified dermatologist.
 
---
 
## Tech stack
 
| Layer | Tool | Purpose |
|---|---|---|
| Neural network | PyTorch + torchvision | Model training and inference |
| Architecture | ResNet18 (pretrained) | Transfer learning backbone |
| Data handling | Pandas + openpyxl | Reading the Excel metadata file |
| Image processing | Pillow + OpenCV | Loading and preprocessing images |
| Evaluation | scikit-learn | Confusion matrix, F1, recall |
| Web backend | Flask | Serving the model as an API |
| Web frontend | HTML + vanilla JS | Upload UI and results chart |
 
---