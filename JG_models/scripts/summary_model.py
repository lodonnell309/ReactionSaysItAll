import sys
import os
import torch
from torchinfo import summary

# Fix path so we can import your model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn_model_big_boi import EmotionCNN

# Initialize model
model = EmotionCNN()

# Generate summary
model_summary = summary(
    model,
    input_size=(1, 1, 48, 48),
    col_names=["input_size", "output_size", "num_params"],
    verbose=0
)

# Print to console
print(model_summary)

# Save to file
os.makedirs("outputs", exist_ok=True)
with open("outputs/emotion_cnn_summary.txt", "w", encoding="utf-8") as f:
    f.write(str(model_summary))

print("✅ Saved model summary to outputs/emotion_cnn_summary.txt")
