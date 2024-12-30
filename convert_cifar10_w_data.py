import os
import sys
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize
import numpy as np


def convert_to_tensor_dataset(root_dir, output_path, img_size=(32, 32)):
    transform = Compose([Resize(img_size), ToTensor()])
    dataset = ImageFolder(root=root_dir, transform=transform)
    data = []
    labels = []
    for img, label in dataset:
        data.append(img.numpy())
        labels.append(label)
    data = torch.tensor(np.stack(data))
    labels = torch.tensor(labels)
    torch.save({'data': data, 'labels': labels}, output_path)
    print(f"Datasets saved at {output_path}, including {len(data)} images.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_tensor.py <input_dir> <output_file>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    convert_to_tensor_dataset(input_dir, output_file)
