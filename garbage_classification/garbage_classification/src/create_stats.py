from pathlib import Path
import json

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

def count_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return 0

    return len([
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


stats = {
    "train_counts": {},
    "test_counts": {}
}


for cls in CLASSES:
    stats["train_counts"][cls] = count_images(
        f"data/train/{cls}"
    )

    stats["test_counts"][cls] = count_images(
        f"data/test/{cls}"
    )


with open("data/dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=4)


print(stats)