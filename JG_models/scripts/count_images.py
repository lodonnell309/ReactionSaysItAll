import os
from collections import defaultdict

def count_images_by_class(base_path):
    counts = defaultdict(dict)

    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split)
        for emotion in sorted(os.listdir(split_path)):
            emotion_dir = os.path.join(split_path, emotion)
            if os.path.isdir(emotion_dir):
                num_images = len([
                    f for f in os.listdir(emotion_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                counts[emotion][split] = num_images

    return counts

# Change this if your data is in a different path
base_dataset_path = "data/fer2013"


image_counts = count_images_by_class(base_dataset_path)

# Print the counts
print("\nImage count by emotion:")
for emotion, splits in image_counts.items():
    train_count = splits.get('train', 0)
    test_count = splits.get('test', 0)
    print(f"{emotion.capitalize():<10} | Train: {train_count:<5} | Test: {test_count:<5}")
