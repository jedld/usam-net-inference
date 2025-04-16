import numpy as np
import torch
import tqdm
import cv2

# Parameters
focal_length = 721.5377  # example focal length in pixels
baseline = 0.54  # example baseline in meters

def compute_d1(estimated_disparity, ground_truth_disparity, threshold=3):
    # Ensure both disparity maps are floats for accurate calculations
    estimated_disparity = estimated_disparity.astype(np.float32)
    ground_truth_disparity = ground_truth_disparity.astype(np.float32)
    
    # Calculate the absolute difference
    disparity_error = np.abs(estimated_disparity - ground_truth_disparity)
    
    # Calculate the number of pixels where the disparity error is above the threshold
    error_pixels = np.sum(disparity_error > threshold)
    
    # Calculate the total number of valid ground truth pixels
    valid_pixels = np.sum(ground_truth_disparity > 0)  # assuming zero or negative values are invalid
    
    # Calculate the D1 loss percentage
    d1_loss = (error_pixels / valid_pixels) * 100 if valid_pixels > 0 else 0
    
    return d1_loss

def compute_epe(predicted_disparity, ground_truth_disparity):
    # For driving stereo dataset we have to mask the zero values
    mask = (ground_truth_disparity > 1e-08).float()
    # count the number of valid pixels
    valid_pixels = mask.sum((-1, -2)).float()
    predicted_disparity = predicted_disparity * (ground_truth_disparity > 0).float()
    return torch.mean(torch.abs(predicted_disparity - ground_truth_disparity).sum((-1, -2)) / valid_pixels).cpu().numpy()

def compute_base_line_and_focal_length(calib_file_path):
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'P_rect_101' in line:
                sline = line.split(' ')
                focal_length = float(sline[1])
            if 'T_101' in line:
                sline = line.split(' ')
                baseline = float(sline[1])

    return focal_length, baseline


def disparity_to_depth(disparity, baseline, focal_length):
    # Ensure disparity has no zeros to avoid division by zero
    disparity[disparity == 0] = 0.1

    focal_length = focal_length.view(-1, 1, 1, 1).expand_as(disparity)
    baseline = baseline.view(-1, 1, 1, 1).expand_as(disparity)

    depth = focal_length * baseline / disparity
    return depth


def calculate_ard_and_gd(predicted_disp, true_disp, depth_true, min_depth, max_depth, r, interval, baseline, focal_length):
    with torch.no_grad():
        ard_values = []
        
        depth_intervals = torch.arange(min_depth, max_depth, interval, device=predicted_disp.device)
        depth_true.to(predicted_disp.device)

        for k in depth_intervals:
            # Apply mask for each image in the batch
            mask = (depth_true > 0) & (depth_true >= (k - r)) & (depth_true <= (k + r))
            
            # get total number of masked pixels, sum the mask along the last three dimensions
            n = mask.sum((-1, -2)).float() + 1e-8

            ard = (torch.abs((predicted_disp - true_disp) / (true_disp + 1e-8)) * mask).sum((-1, -2)) / n
            ard_values.append(ard)

        # Compute mean ARD for the batch if not empty
        ard_values = torch.stack(ard_values)
        ard_values = ard_values.mean(1)
        gd = ard_values.mean()
        return ard_values, gd

def evaluate_seg(model, dataloader, device, min_depth = 0, max_depth = 80, r = 4, interval = 8, title='Seg'):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()
    best_image = None
    worst_image = None
    best_gd = 1000
    worst_gd = 0
    with torch.no_grad():
        total_test_loss = 0
        total_ard_values = 0
        total_gd = 0
        total_epe = 0.0
        total_d1 = 0.0
        for left_image, right_image, left_mask, disparity_map, depth_map, focal_length, baseline in tqdm.tqdm(dataloader):
            left_image = left_image.to(device)
            right_image = right_image.to(device)
            left_mask = left_mask.to(device)
            disparity_map = disparity_map.to(device)
            focal_length = focal_length.to(device)
            baseline = baseline.to(device)
            depth_map = depth_map.to(device)

            outputs = model(torch.cat((left_image, right_image, left_mask), 1))
            mask = disparity_map > (0 + 1e-8)
            outputs = outputs * mask.float()
            test_loss = criterion(outputs, disparity_map)
            
            total_test_loss += test_loss.item()

            ard_values, gd = calculate_ard_and_gd(outputs, disparity_map, depth_map, min_depth, max_depth, r, interval, baseline, focal_length)
            total_epe += compute_epe(outputs, disparity_map)
            total_d1 += compute_d1(outputs.cpu().numpy(), disparity_map.cpu().numpy())
            total_ard_values += ard_values
            total_gd += gd

            # get the index of the best gd in this batch
            if best_image is None or (gd < best_gd):
                best_gd = gd
                best_image = (left_image, outputs, disparity_map, depth_map, gd)

            if worst_image is None or (gd > worst_gd):
                worst_gd = gd
                worst_image = (left_image, outputs, disparity_map, depth_map, gd)

    # write best and worst images to disk
    left_img = best_image[0][0].cpu().numpy()
    left_img = left_img.transpose(1, 2, 0)
    left_img = (left_img * 255).astype(np.uint8)

    pred_disp = best_image[1][0].cpu().numpy().squeeze()
    gt_disp = best_image[2][0].cpu().numpy().squeeze()

    # enhance by making the historgram uniform
    pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
    pred_disp = (pred_disp * 255).astype(np.uint8)

    gt_disp = (gt_disp - gt_disp.min()) / (gt_disp.max() - gt_disp.min())
    gt_disp = (gt_disp * 255).astype(np.uint8)

    cv2.imwrite(f'best_{title}_image_pred_disp.png', pred_disp)
    cv2.imwrite(f'best_{title}_image_gt_disp.png', gt_disp)
    cv2.imwrite(f'best_{title}_image.png', left_img)

    test_loss = total_test_loss/len(dataloader)
    total_ard = total_ard_values/len(dataloader)
    total_gd = total_gd/len(dataloader)
    total_epe = total_epe/len(dataloader)
    total_d1 = total_d1/len(dataloader)

    return test_loss, total_ard, total_gd.cpu().numpy(), total_epe, total_d1

def evaluate_baseline(model, dataloader, device, min_depth = 0, max_depth = 80, r = 4, interval = 8, title='Baseline'):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        total_ard_values = 0
        total_gd = 0
        best_image = None
        worst_image = None
        best_gd = 1000
        worst_gd = 0
        total_epe = 0.0
        total_d1 = 0.0
        for left_image, right_image, disparity_map, depth_map, focal_length, baseline in tqdm.tqdm(dataloader):
            left_image = left_image.to(device)
            right_image = right_image.to(device)
            disparity_map = disparity_map.to(device)
            focal_length = focal_length.to(device)
            baseline = baseline.to(device)
            depth_map = depth_map.to(device)

            outputs = model(torch.cat((left_image, right_image), 1))
            mask = disparity_map > 1e-8
            outputs = outputs * mask.float()
            test_loss = criterion(outputs, disparity_map)
            
            total_test_loss += test_loss.item()

            ard_values, gd = calculate_ard_and_gd(outputs, disparity_map, depth_map, min_depth, max_depth, r, interval, baseline, focal_length)
            total_epe += compute_epe(outputs, disparity_map)
            total_d1 += compute_d1(outputs.cpu().numpy(), disparity_map.cpu().numpy())
            total_ard_values += ard_values
            total_gd += gd

            # get the index of the best gd in this batch
            if best_image is None or (gd < best_gd):
                best_gd = gd
                best_image = (left_image, outputs, disparity_map, depth_map, gd)

            if worst_image is None or (gd > worst_gd):
                worst_gd = gd
                worst_image = (left_image, outputs, disparity_map, depth_map, gd)

    # write best and worst images to disk
    left_img = best_image[0][0].cpu().numpy()
    left_img = left_img.transpose(1, 2, 0)
    left_img = (left_img * 255).astype(np.uint8)

    pred_disp = best_image[1][0].cpu().numpy().squeeze()
    gt_disp = best_image[2][0].cpu().numpy().squeeze()

    # enhance by making the historgram uniform
    pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
    pred_disp = (pred_disp * 255).astype(np.uint8)

    gt_disp = (gt_disp - gt_disp.min()) / (gt_disp.max() - gt_disp.min())
    gt_disp = (gt_disp * 255).astype(np.uint8)

    cv2.imwrite(f'best_{title}_image_pred_disp.png', pred_disp)
    cv2.imwrite(f'best_{title}_image_gt_disp.png', gt_disp)
    cv2.imwrite(f'best_{title}_image.png', left_img)
    print(f"len: {len(dataloader)}")
    test_loss = total_test_loss/len(dataloader)
    total_ard = total_ard_values/len(dataloader)
    total_gd = total_gd/len(dataloader)
    total_epe = total_epe/len(dataloader)
    total_d1 = total_d1/len(dataloader)

    return test_loss, total_ard, total_gd.cpu().numpy(), total_epe, total_d1

# # Example usage
# predicted_disp = np.random.rand(480, 640) * 10  # Example predicted disparity map
# true_disp = np.random.rand(480, 640) * 10  # Example ground truth disparity map
# min_depth = 0  # Minimum depth in meters
# max_depth = 80  # Maximum depth in meters, adjust as necessary
# r = 4  # Measuring range in meters
# interval = 8  # Sampling interval in meters

# ard_values, gd = calculate_ard_and_gd(predicted_disp, true_disp, min_depth, max_depth, r, interval)
# print("ARD values:", ard_values)
# print("Global Difference (GD):", gd)
