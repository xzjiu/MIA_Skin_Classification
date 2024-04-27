import os
import json
from PIL import Image
from torch.utils.data import Dataset

class SkinLesionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Iterate through each folder in the root directory
        for patient_dir in os.listdir(root_dir):
            img_path = os.path.join(root_dir, patient_dir, f"{patient_dir}_Dermoscopic_Image")
            mask_path = os.path.join(root_dir, patient_dir, f"{patient_dir}_lesion")
            json_path = os.path.join(root_dir, patient_dir, f"{patient_dir}.json")  # JSON file path

            if os.path.exists(json_path):
                with open(json_path, 'r') as file:
                    labels = json.load(file)
            else:
                labels = {'Lesion Class': 'Unknown', 'Asymmetry Label': 'Unknown'}

            # Check if directories exist and add them along with labels
            if os.path.isdir(img_path) and os.path.isdir(mask_path):
                img_files = [f for f in os.listdir(img_path) if f.endswith('.bmp')]
                mask_files = [f for f in os.listdir(mask_path) if f.endswith('.bmp')]
                if img_files and mask_files:  # Ensure there is a corresponding mask file
                    self.samples.append({
                        'image_path': os.path.join(img_path, img_files[0]),
                        'mask_path': os.path.join(mask_path, mask_files[0]),
                        'lesion_class': labels['Lesion Class'],
                        'asymmetry_label': labels['Asymmetry Label']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        mask = Image.open(sample['mask_path']).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {
            'image': image,
            'mask': mask,
            'lesion_class': sample['lesion_class'],
            'asymmetry_label': sample['asymmetry_label']
        }
