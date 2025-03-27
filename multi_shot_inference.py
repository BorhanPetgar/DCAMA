# multi_shot_inference.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import glob

from model.DCAMA import DCAMA
from common import utils


def parse_args():
    parser = argparse.ArgumentParser(description='DCAMA Single Image Inference')
    parser.add_argument('--backbone', type=str, default='swin')
    parser.add_argument('--feature_extractor_path', type=str, required=True)
    parser.add_argument('--load', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--support_img', type=str, required=True, help='Path to support image or directory with multiple images')
    parser.add_argument('--support_mask', type=str, required=True, help='Path to support mask or directory with multiple masks')
    parser.add_argument('--query_img', type=str, required=True)
    parser.add_argument('--output_mask', type=str, required=True)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--duplicate_support', action='store_true', help='Duplicate single support instead of using multiple')
    return parser.parse_args()


def load_image(path, is_mask=False, img_size=384):
    if is_mask:
        img = Image.open(path).convert('L')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        # Binary mask - threshold to ensure it's exactly 0 and 1
        img_tensor = transform(img)
        img_tensor = (img_tensor > 0.5).float()
        return img_tensor.squeeze(0)
    else:
        img = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img)


def load_support_images(img_path, mask_path, is_dir, nshot, img_size):
    """Load support images and masks, either multiple from directory or duplicate single ones"""
    if is_dir:
        # Load multiple support images from directory
        img_paths = sorted(glob.glob(os.path.join(img_path, '*.jpg')))[:nshot] + \
                   sorted(glob.glob(os.path.join(img_path, '*.png')))[:nshot]
        mask_paths = sorted(glob.glob(os.path.join(mask_path, '*.png')))[:nshot]
        
        # Ensure we have enough images
        if len(img_paths) < nshot or len(mask_paths) < nshot:
            print(f"Warning: Found only {len(img_paths)} images and {len(mask_paths)} masks, but nshot={nshot}")
            nshot = min(len(img_paths), len(mask_paths))
        
        # Load all images and masks
        support_imgs = []
        support_masks = []
        for idx in range(nshot):
            img = load_image(img_paths[idx], False, img_size)
            mask = load_image(mask_paths[idx], True, img_size)
            support_imgs.append(img)
            support_masks.append(mask)
        
        # Stack them into tensors
        support_imgs = torch.stack(support_imgs, dim=0)  # [nshot, 3, H, W]
        support_masks = torch.stack(support_masks, dim=0)  # [nshot, 1, H, W]
    else:
        # Load single support image and duplicate
        img = load_image(img_path, False, img_size).unsqueeze(0)  # [1, 3, H, W]
        mask = load_image(mask_path, True, img_size).unsqueeze(0)  # [1, 1, H, W]
        
        # Duplicate if needed
        if nshot > 1:
            support_imgs = img.repeat(nshot, 1, 1, 1)  # [nshot, 3, H, W]
            support_masks = mask.repeat(nshot, 1, 1, 1)  # [nshot, 1, H, W]
        else:
            support_imgs = img
            support_masks = mask
    
    return support_imgs, support_masks


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = DCAMA(args.backbone, args.feature_extractor_path, args.use_original_imgsize)
    model.eval()
    model = nn.DataParallel(model)
    model.to(device)
    
    # Load trained model
    params = model.state_dict()
    state_dict = torch.load(args.load, map_location=device)
    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)
    model.load_state_dict(state_dict)
    
    # Determine if we're using multiple support images or duplicating
    is_dir = os.path.isdir(args.support_img) and os.path.isdir(args.support_mask) and not args.duplicate_support
    
    # Load support images and masks (either multiple different ones or duplicates)
    support_imgs, support_masks = load_support_images(
        args.support_img, args.support_mask, is_dir, args.nshot, args.img_size
    )
    
    # Load query image
    query_img = load_image(args.query_img, False, args.img_size).unsqueeze(0)  # [1, 3, H, W]
    
    # Prepare batch - ensure correct dimensions
    batch = {
        'support_imgs': support_imgs.unsqueeze(0),  # [1, nshot, 3, H, W]
        'support_masks': support_masks.unsqueeze(0),  # [1, nshot, 1, H, W]
        'query_img': query_img,  # [1, 3, H, W]
        'query_mask': torch.zeros((1, 1, args.img_size, args.img_size)),  # Placeholder
        'class_id': torch.tensor([0])  # Placeholder
    }
    
    # Print shapes for debugging
    print(f"Support images shape: {batch['support_imgs'].shape}")
    print(f"Support masks shape: {batch['support_masks'].shape}")
    print(f"Query image shape: {batch['query_img'].shape}")
    
    # Move batch to device
    batch = utils.to_cuda(batch)
    
    # Perform inference
    with torch.no_grad():
        pred_mask = model.module.predict_mask_nshot(batch, nshot=args.nshot)
        print(20 * "=")
        print(f'Prediction shape: {pred_mask.shape}')
        
        # Debug prediction values
        print(f"Prediction values - Min: {pred_mask.min().item()}, Max: {pred_mask.max().item()}")
        print(f"Prediction unique values: {torch.unique(pred_mask).cpu().numpy()}")
        
    # Process the prediction
    pred_np = pred_mask[0].cpu().numpy() if pred_mask.dim() == 3 else pred_mask[0, 0].cpu().numpy()
    
    # Visualize as heatmap
    pred_viz = ((pred_np - pred_np.min()) / max(1e-8, pred_np.max() - pred_np.min()) * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_viz)
    
    # If not using original image size, resize back to original query image size
    if not args.use_original_imgsize:
        original_size = Image.open(args.query_img).size  # (W, H)
        output_img = output_img.resize((original_size[0], original_size[1]), Image.NEAREST)
    
    # Save the result
    output_img.save(args.output_mask)
    print(f"Prediction saved to {args.output_mask}")


if __name__ == '__main__':
    main()