from torch.utils.data import Dataset
import cv2
import os

class CalibrationStereoDataset(Dataset):
    def __init__(self, left_dir, right_dir, transform=None):
        self.left_images = sorted(os.listdir(left_dir))
        self.right_images = sorted(os.listdir(right_dir))
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_path = os.path.join(self.left_dir, self.left_images[idx])
        right_path = os.path.join(self.right_dir, self.right_images[idx])

        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return left_img, right_img
