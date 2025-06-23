import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotation_file, tokenize, transform=None):
        self.image_folder = image_folder
        self.annotations = self.load_annotations(annotation_file)
        self.transform = transform
        self.tokenize = tokenize

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, annotation = self.annotations[idx]
        annotation = self.tokenize(annotation)
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, annotation

    def load_annotations(self, annotation_file):
        with open(annotation_file, "r") as f:
            lines = f.readlines()
        annotations = [line.strip().split(", ") for line in lines]
        return annotations
