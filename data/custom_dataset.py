import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomFSSDataset(Dataset):
    """Custom Few-Shot Segmentation Dataset"""
    
    def __init__(self, train_img_dir, train_mask_dir, test_img_dir=None, test_mask_dir=None, 
                 mode='train', img_size=384, use_original_imgsize=False, n_shots=1):
        """
        Args:
            train_img_dir (str): Directory with all training images
            train_mask_dir (str): Directory with all training masks
            test_img_dir (str): Directory with all test images (optional)
            test_mask_dir (str): Directory with all test masks (optional)
            mode (str): 'train' or 'test'
            img_size (int): Target image size
            use_original_imgsize (bool): If True, use original image size during evaluation
            n_shots (int): Number of support examples
        """
        self.train_img_dir = train_img_dir
        self.train_mask_dir = train_mask_dir
        self.test_img_dir = test_img_dir if test_img_dir else train_img_dir
        self.test_mask_dir = test_mask_dir if test_mask_dir else train_mask_dir
        self.mode = mode
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.n_shots = n_shots
        
        # Get image file names
        self.train_img_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if test_img_dir:
            self.test_img_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.test_img_files = self.train_img_files
            
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Set up for evaluation mode
        if self.use_original_imgsize and mode == 'test':
            self.transform_original = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform_original = transforms.Compose([
                transforms.ToTensor()
            ])
        
        print(f"Loaded dataset with {len(self.train_img_files)} training images and {len(self.test_img_files)} test images")

    def __len__(self):
        return len(self.train_img_files) if self.mode == 'train' else len(self.test_img_files)

    def __getitem__(self, idx):
        # For training/validation: 
        # - Random query from training images
        # - Random support set from other training images
        if self.mode == 'train':
            query_img_path = os.path.join(self.train_img_dir, self.train_img_files[idx])
            query_mask_path = os.path.join(self.train_mask_dir, self.train_img_files[idx])
            
            # Handle case where mask has different filename extension
            if not os.path.exists(query_mask_path):
                base_name = os.path.splitext(self.train_img_files[idx])[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(self.train_mask_dir, base_name + ext)
                    if os.path.exists(test_path):
                        query_mask_path = test_path
                        break
            
            # Get random support samples (different from query)
            support_idxs = []
            while len(support_idxs) < self.n_shots:
                support_idx = random.randint(0, len(self.train_img_files) - 1)
                if support_idx != idx and support_idx not in support_idxs:
                    support_idxs.append(support_idx)
        else:
            # For testing: 
            # - Query from test set
            # - Support set from training set
            query_img_path = os.path.join(self.test_img_dir, self.test_img_files[idx])
            query_mask_path = os.path.join(self.test_mask_dir, self.test_img_files[idx])
            
            # Handle case where mask has different filename extension
            if not os.path.exists(query_mask_path):
                base_name = os.path.splitext(self.test_img_files[idx])[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(self.test_mask_dir, base_name + ext)
                    if os.path.exists(test_path):
                        query_mask_path = test_path
                        break
            
            # Get random support samples from training set
            support_idxs = random.sample(range(len(self.train_img_files)), self.n_shots)
        
        # Load query image and mask
        query_img = Image.open(query_img_path).convert('RGB')
        query_mask = Image.open(query_mask_path).convert('L')
        
        # Load support images and masks
        support_imgs = []
        support_masks = []
        for support_idx in support_idxs:
            support_img_path = os.path.join(self.train_img_dir, self.train_img_files[support_idx])
            support_mask_path = os.path.join(self.train_mask_dir, self.train_img_files[support_idx])
            
            # Handle case where mask has different filename extension
            if not os.path.exists(support_mask_path):
                base_name = os.path.splitext(self.train_img_files[support_idx])[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(self.train_mask_dir, base_name + ext)
                    if os.path.exists(test_path):
                        support_mask_path = test_path
                        break
            
            support_img = Image.open(support_img_path).convert('RGB')
            support_mask = Image.open(support_mask_path).convert('L')
            
            support_imgs.append(self.transform(support_img))
            # Convert binary mask (0,255) to binary mask (0,1)
            mask = self.mask_transform(support_mask)
            mask = (mask > 0.5).float()
            support_masks.append(mask)
        
        # Transform query
        query_img = self.transform(query_img)
        # Convert binary mask (0,255) to binary mask (0,1)
        query_mask = self.mask_transform(query_mask)
        query_mask = (query_mask > 0.5).float()
        
        # Also keep original size if needed
        if self.use_original_imgsize and self.mode == 'test':
            query_img_original = Image.open(query_img_path).convert('RGB')
            query_mask_original = Image.open(query_mask_path).convert('L')
            
            query_img_original = self.transform_original(query_img_original)
            query_mask_original = self.mask_transform_original(query_mask_original)
            query_mask_original = (query_mask_original > 0.5).float()
        
        # Stack support images and masks - FIXED SHAPE FORMAT
        support_imgs = torch.stack(support_imgs) # Shape: [n_shots, C, H, W]
        support_masks = torch.stack(support_masks) # Shape: [n_shots, 1, H, W]
        
        # Create sample with proper dimensions for the model
        sample = {
            'query_img': query_img,  # Shape: [C, H, W]
            'query_mask': query_mask.long(),  # Shape: [1, H, W]
            'support_imgs': support_imgs,  # Shape: [n_shots, C, H, W]
            'support_masks': support_masks,  # Shape: [n_shots, 1, H, W]
            'class_id': torch.tensor(1)  # Assume single class for simplicity
        }
        
        if self.use_original_imgsize and self.mode == 'test':
            sample['query_img_original'] = query_img_original
            sample['query_mask_original'] = query_mask_original
        
        return sample

def get_custom_dataloader(train_img_dir, train_mask_dir, test_img_dir=None, test_mask_dir=None,
                          batch_size=4, num_workers=4, n_shots=1, img_size=384, use_original_imgsize=False):
    """Create dataloaders for custom dataset"""
    
    # Training dataloader
    train_dataset = CustomFSSDataset(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        test_img_dir=test_img_dir,
        test_mask_dir=test_mask_dir,
        mode='train',
        img_size=img_size,
        use_original_imgsize=use_original_imgsize,
        n_shots=n_shots
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation/Test dataloader
    val_dataset = CustomFSSDataset(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        test_img_dir=test_img_dir,
        test_mask_dir=test_mask_dir,
        mode='test',
        img_size=img_size,
        use_original_imgsize=use_original_imgsize,
        n_shots=n_shots
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader