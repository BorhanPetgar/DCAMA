import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.DCAMA import DCAMA
from common import utils


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to namespace for compatibility with the rest of the code
    class Config:
        pass
    
    args = Config()
    for key, value in config.items():
        setattr(args, key, value)
    
    return args


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
        return img_tensor.squeeze(0)  # Remove channel dim, return [H, W]
    else:
        img = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img)


def load_support_images(folder_path, img_size=384):
    """Load all support images and their masks from a folder.
    
    Expected format:
    - Images: name.jpg/png
    - Masks: name_mask.png
    """
    support_imgs = []
    support_masks = []
    
    # Get all image files (excluding mask files)
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) 
                and any(f.lower().endswith(ext) for ext in img_extensions)
                and not f.lower().endswith('_mask.png')]
    
    for img_file in img_files:
        img_path = os.path.join(folder_path, img_file)
        
        # Determine mask path (assuming naming convention: image.jpg -> image_mask.png)
        mask_name = os.path.splitext(img_file)[0] + '_mask.png'
        mask_path = os.path.join(folder_path, mask_name)
        
        if os.path.exists(mask_path):
            img = load_image(img_path, False, img_size)
            mask = load_image(mask_path, True, img_size)
            
            support_imgs.append(img)
            support_masks.append(mask)
            print(f"Loaded support pair: {img_file} and {mask_name}")
    
    if not support_imgs:
        raise ValueError(f"No valid support image-mask pairs found in {folder_path}")
    
    # Stack all images and masks into tensors
    support_imgs = torch.stack(support_imgs)  # [nshot, 3, H, W]
    support_masks = torch.stack(support_masks)  # [nshot, H, W]
    
    return support_imgs, support_masks


def main():
    # Load configuration from fixed YAML path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../params/inference.yaml'))
    print(f"Loading configuration from: {config_path}")
    args = load_config(config_path)
    
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
    
    # Load support images from folder
    support_imgs, support_masks = load_support_images(args.support_folder, args.img_size)
    
    # Determine nshot based on the number of support images
    nshot = support_imgs.shape[0]
    print(f"Found {nshot} support images in {args.support_folder}")
    
    # Load query image
    query_img = load_image(args.query_img, False, args.img_size).unsqueeze(0)  # [1, 3, H, W]
    
    # Prepare batch
    batch = {
        'support_imgs': support_imgs.unsqueeze(0),  # [1, nshot, 3, H, W]
        'support_masks': support_masks.unsqueeze(0),  # [1, nshot, H, W]
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
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        print(20 * "=")
        print(f'Prediction shape: {pred_mask.shape}')
        
        # Debug prediction values
        print(f"Prediction values - Min: {pred_mask.min().item()}, Max: {pred_mask.max().item()}")
        print(f"Prediction unique values: {torch.unique(pred_mask).cpu().numpy()}")
        
    # Process the prediction
    pred_np = pred_mask[0, 0].cpu().numpy() if pred_mask.dim() > 3 else pred_mask[0].cpu().numpy()
    
    # Normalize to 0-255 for visualization
    pred_viz = ((pred_np - pred_np.min()) / max(1e-8, pred_np.max() - pred_np.min()) * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_viz)
    
    # If not using original image size, resize back to original query image size
    if not args.use_original_imgsize:
        original_size = Image.open(args.query_img).size  # (W, H)
        output_img = output_img.resize(original_size, Image.NEAREST)
    
    # Save the result
    output_img.save(args.output_mask)
    print(f"Prediction saved to {args.output_mask}")


if __name__ == '__main__':
    main()