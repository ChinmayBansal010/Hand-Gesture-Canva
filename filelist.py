import os
import json

DATA_DIR = 'data'
FILELIST = os.path.join(DATA_DIR, 'filelist.txt')
TXT_OUTPUT = os.path.join(DATA_DIR, 'class_names.txt')
JSON_OUTPUT = os.path.join(DATA_DIR, 'label_to_idx.json')

def generate_label_to_idx(filelist_path):
    samples = []
    with open(filelist_path, 'r') as f:
        for line in f:
            path = line.strip()
            label = os.path.basename(os.path.dirname(path))
            samples.append(label)

    unique_labels = sorted(set(samples))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_idx, unique_labels

if __name__ == "__main__":
    if not os.path.exists(FILELIST):
        print(f"Filelist not found: {FILELIST}")
        exit(1)

    label_to_idx, class_names = generate_label_to_idx(FILELIST)

    # Save to TXT
    with open(TXT_OUTPUT, 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    print(f"Saved class names to {TXT_OUTPUT}")

    # Optional: Save mapping to JSON
    with open(JSON_OUTPUT, 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    print(f"Saved label_to_idx mapping to {JSON_OUTPUT}")
