import os
import torch
from collections import Counter

def count_class_distribution(data_dir):
    label_counts = Counter()
    files = sorted(os.listdir(data_dir))
    print(f"📂 Scanning {len(files)} files in: {data_dir}")

    for fname in files:
        sample = torch.load(os.path.join(data_dir, fname), weights_only=True)
        label = sample['label']
        label_counts[int(label)] += 1

    print("\n📊 Class distribution:")
    for cls, count in sorted(label_counts.items()):
        print(f"Class {cls}: {count} samples")

if __name__ == "__main__":
    count_class_distribution("data/fer2013_pt/train")
