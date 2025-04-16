# create a train loader that loads the images and disparity maps

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def compute_base_line_and_focal_length(calib_file_path, cache = {}):
    if calib_file_path in cache:
        return cache[calib_file_path]
    
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f'Calibration file {calib_file_path} not found')
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'P_rect_101' in line:
                sline = line.split(' ')
                focal_length = float(sline[1])
            if 'T_103' in line:
                sline = line.split(' ')
                baseline = float(sline[1])
    
    cache[calib_file_path] = focal_length, baseline

    return focal_length, baseline
class StereoSegmentationDataset(Dataset):
    def __init__(self, left_images_folder, right_images_folder,
                 left_masks_folder,
                 disparity_maps_folder,
                 depth_map_folder,
                 calibration_folder,
                 sky_mask_folder=None,
                 randomFlip=False,
                 transform=None, 
                 transform_segmentaton=None,
                 transform_disparity=None):
        self.left_images_folder = left_images_folder
        self.right_images_folder = right_images_folder
        self.left_masks_folder = left_masks_folder
        # self.right_masks_folder = right_masks_folder
        self.disparity_maps_folder = disparity_maps_folder
        self.depth_map_folder = depth_map_folder
        self.transform = transform
        self.transform_segmentation = transform_segmentaton
        self.transform_disparity = transform_disparity
        self.randomFlip = randomFlip
        self.sky_mask_folder = sky_mask_folder
        
        self.left_images = os.listdir(left_images_folder)
        self.left_masks = os.listdir(left_masks_folder)
        # self.right_masks_folder = os.listdir(right_masks_folder)
        self.right_images = os.listdir(right_images_folder)
        self.disparity_maps = os.listdir(disparity_maps_folder)
        self.depth_maps = os.listdir(depth_map_folder)
        if sky_mask_folder:
            self.sky_masks = os.listdir(sky_mask_folder)
        
        self.left_images.sort()
        self.right_images.sort()
        self.left_masks.sort()
        # self.right_masks_folder.sort()
        self.disparity_maps.sort()
        self.depth_maps.sort()
        if sky_mask_folder:
            self.sky_masks.sort()

        self.calibration_folder = calibration_folder
        self.calibration_cache = {}

        # check that the number of images is the same
        print(f'Found {len(self.left_images)} left images')
        print(f'Found {len(self.right_images)} right images')
        print(f'Found {len(self.disparity_maps)} disparity maps')
        print(f'Found {len(self.left_masks)} left masks')
        print(f'Found {len(self.depth_maps)} depth maps')
        if sky_mask_folder:
            print(f'Found {len(self.sky_masks)} sky masks')
        
        assert len(self.left_images) == len(self.right_images) == len(self.disparity_maps) == len(self.left_masks) == len(self.depth_maps)
        if sky_mask_folder:
            assert len(self.left_images) == len(self.sky_masks)

    def __len__(self):
        return len(self.left_images)
    
    def __getitem__(self, idx):
        left_image = cv2.imread(os.path.join(self.left_images_folder, self.left_images[idx]))
        right_image = cv2.imread(os.path.join(self.right_images_folder, self.right_images[idx]))
        left_mask = cv2.imread(os.path.join(self.left_masks_folder, self.left_masks[idx]))
        # right_mask = cv2.imread(os.path.join(self.right_masks_folder, self.right_masks_folder[idx]))
        disparity_map = cv2.imread(os.path.join(self.disparity_maps_folder, self.disparity_maps[idx]), cv2.IMREAD_UNCHANGED)
        depth_map = cv2.imread(os.path.join(self.depth_map_folder, self.depth_maps[idx]), cv2.IMREAD_UNCHANGED)
        if self.sky_mask_folder:
            sky_mask = cv2.imread(os.path.join(self.sky_mask_folder, self.sky_masks[idx]), cv2.IMREAD_UNCHANGED)
        
        calibration_filename = self.left_images[idx].split('_')[0] + '.txt'
        focal_length, baseline = compute_base_line_and_focal_length(os.path.join(self.calibration_folder, calibration_filename), self.calibration_cache)

        # copy the bottom 1/4 of the image to the top 1/4
        if self.randomFlip:
            select = torch.rand(1)
            # copy the bottom 1/4 of the image to the top 1/4
            if select > 0.8:
                height, _ = left_image.shape[:2]
                left_image[:height//4] = left_image[3*height//4:]
                right_image[:height//4] = right_image[3*height//4:]
                left_mask[:height//4] = left_mask[3*height//4:]
                disparity_map[:height//4] = disparity_map[3*height//4:]
                depth_map[:height//4] = depth_map[3*height//4:]

                # don't use sky masks
                if self.sky_mask_folder:
                    sky_mask = np.zeros(sky_mask.shape, dtype=sky_mask.dtype)

            # copy the bottom 3/4 of the image to the top 1/4
            elif select > 0.6:
                height, _ = left_image.shape[:2]
                left_image[:height//4] = left_image[2*height//4:3*height//4]
                right_image[:height//4] = right_image[2 * height//4:3*height//4]
                left_mask[:height//4] = left_mask[2*height//4:3*height//4]
                disparity_map[:height//4] = disparity_map[2 * height//4:3*height//4]
                depth_map[:height//4] = depth_map[2 *height//4:3*height//4]

                # don't use sky masks
                if self.sky_mask_folder:
                    sky_mask = np.zeros(sky_mask.shape, dtype=sky_mask.dtype)              

        left_image = self.transform(left_image)
        right_image = self.transform(right_image)
        left_mask = self.transform_segmentation(left_mask)

        disparity_map = self.transform_disparity(disparity_map)
        depth_map = self.transform_disparity(depth_map)
        if self.sky_mask_folder:
            sky_mask = self.transform_disparity(sky_mask)
        
        focal_length = torch.tensor(focal_length).unsqueeze(0)
        baseline = torch.tensor(baseline).unsqueeze(0)
        if self.sky_mask_folder:
            return left_image, right_image, left_mask, disparity_map, depth_map, focal_length, baseline, sky_mask
        
        return left_image, right_image, left_mask, disparity_map, depth_map, focal_length, baseline
class StereoDataset(Dataset):
    def __init__(self, left_images_folder, right_images_folder, disparity_maps_folder, depth_maps_folder, calibration_folder, sky_mask_folder=None, randomFlip=False, transform=None, transform_disparity=None):
        self.left_images_folder = left_images_folder
        self.right_images_folder = right_images_folder
        self.disparity_maps_folder = disparity_maps_folder
        self.depth_maps_folder = depth_maps_folder
        self.transform = transform
        self.transform_disparity = transform_disparity
        self.calibration_folder = calibration_folder
        self.sky_mask_folder = sky_mask_folder
        self.calibration_cache = {}
        self.randomFlip = randomFlip
        
        self.left_images = os.listdir(left_images_folder)
        self.right_images = os.listdir(right_images_folder)
        self.disparity_maps = os.listdir(disparity_maps_folder)
        self.depth_maps = os.listdir(depth_maps_folder)

        if sky_mask_folder:
            self.sky_masks = os.listdir(sky_mask_folder)
        
        self.left_images.sort()
        self.right_images.sort()
        self.disparity_maps.sort()
        self.depth_maps.sort()
        if sky_mask_folder:
            self.sky_masks.sort()

        # check that the number of images is the same
        print(f'Found {len(self.left_images)} left images')
        print(f'Found {len(self.right_images)} right images')
        print(f'Found {len(self.disparity_maps)} disparity maps')
        print(f'Found {len(self.depth_maps)} depth maps')
        if sky_mask_folder:
            print(f'Found {len(self.sky_masks)} sky masks')
        
        assert len(self.left_images) == len(self.right_images) == len(self.disparity_maps) == len(self.depth_maps)
        if sky_mask_folder:
            assert len(self.left_images) == len(self.sky_masks)

        
    def __len__(self):
        return len(self.left_images)
    
    def __getitem__(self, idx):
        left_image = cv2.imread(os.path.join(self.left_images_folder, self.left_images[idx]))
        right_image = cv2.imread(os.path.join(self.right_images_folder, self.right_images[idx]))
        disparity_map = cv2.imread(os.path.join(self.disparity_maps_folder, self.disparity_maps[idx]), cv2.IMREAD_UNCHANGED)
        depth_map = cv2.imread(os.path.join(self.depth_maps_folder, self.depth_maps[idx]), cv2.IMREAD_UNCHANGED)
        if self.sky_mask_folder:
            sky_mask = cv2.imread(os.path.join(self.sky_mask_folder, self.sky_masks[idx]), cv2.IMREAD_UNCHANGED)
        
        calibration_filename = self.left_images[idx].split('_')[0] + '.txt'
        focal_length, baseline = compute_base_line_and_focal_length(os.path.join(self.calibration_folder, calibration_filename), self.calibration_cache)

        

        if self.randomFlip:
            select = torch.rand(1)

            # copy the bottom 1/4 of the image to the top 1/4
            if select > 0.8:
                height, _ = left_image.shape[:2]
                left_image[:height//4] = left_image[3*height//4:]
                right_image[:height//4] = right_image[3*height//4:]
                disparity_map[:height//4] = disparity_map[3*height//4:]
                depth_map[:height//4] = depth_map[3*height//4:]

                # don't use sky masks
                if self.sky_mask_folder:
                    sky_mask = np.zeros(sky_mask.shape, dtype=sky_mask.dtype)

            # copy the bottom 3/4 of the image to the top 1/4
            elif select > 0.6:
                height, _ = left_image.shape[:2]
                left_image[:height//4] = left_image[2*height//4:3*height//4]
                right_image[:height//4] = right_image[2 * height//4:3*height//4]
                disparity_map[:height//4] = disparity_map[2 * height//4:3*height//4]
                depth_map[:height//4] = depth_map[2 *height//4:3*height//4]

                # don't use sky masks
                if self.sky_mask_folder:
                    sky_mask = np.zeros(sky_mask.shape, dtype=sky_mask.dtype)
            


        left_image = self.transform(left_image)
        right_image = self.transform(right_image)
        disparity_map = self.transform_disparity(disparity_map)
        depth_map = self.transform_disparity(depth_map)
        if self.sky_mask_folder:
            sky_mask = self.transform_disparity(sky_mask)
        focal_length = torch.tensor(focal_length).unsqueeze(0)
        baseline = torch.tensor(baseline).unsqueeze(0)

        if self.sky_mask_folder:
            return left_image, right_image, disparity_map, depth_map, focal_length, baseline, sky_mask
        
        return left_image, right_image, disparity_map, depth_map, focal_length, baseline