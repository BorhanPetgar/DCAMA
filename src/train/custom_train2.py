r""" Custom training code for DCAMA """
import os
import torch.optim as optim
import torch
import time
from datetime import datetime
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.DCAMA import DCAMA
from common.evaluation import Evaluator
from common import utils
# Import your custom dataset
from data.custom_dataset import get_custom_dataloader


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


def train(epoch, model, dataloader, optimizer, training):
    r""" Train """
    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.train_mode() if training else model.eval()
    
    total_loss = 0.0
    total_inter, total_union = 0, 0
    samples_count = 0
    
    # For progress reporting
    start_time = time.time()
    phase = 'Training' if training else 'Validation'
    
    for idx, batch in enumerate(dataloader):
        # 1. forward pass
        batch = utils.to_cuda(batch)
        
        # Debug shapes before proceeding
        if idx == 0:
            print(f"Query image shape: {batch['query_img'].shape}")
            print(f"Support images shape: {batch['support_imgs'].shape}")
            print(f"Support masks shape: {batch['support_masks'].shape}")
            print(f"Query mask shape: {batch['query_mask'].shape}")
        
        # Reshape the support tensors to match what the model expects
        # Reshape [B, 1, C, H, W] to [B, C, H, W]
        support_imgs = batch['support_imgs'].squeeze(1)
        # Reshape [B, 1, 1, H, W] to [B, H, W]
        support_masks = batch['support_masks'].squeeze(1).squeeze(1)
        
        if idx == 0:
            print(f"Reshaped support images: {support_imgs.shape}")
            print(f"Reshaped support masks: {support_masks.shape}")
        
        # Pass reshaped tensors to the model
        logit_mask = model(batch['query_img'], support_imgs, support_masks)
        pred_mask = logit_mask.argmax(dim=1)
        
        if idx == 0:
            print(f"Logit mask shape: {logit_mask.shape}")
            print(f"Pred mask shape: {pred_mask.shape}")
            print(f"Query mask shape: {batch['query_mask'].shape}")
            print(f"Unique values in pred_mask: {torch.unique(pred_mask)}")
            print(f"Unique values in query_mask: {torch.unique(batch['query_mask'])}")
        
        # 2. Compute loss & update model parameters
        loss = model.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction - DIY version to fix dimensionality issues
        # Extract query mask and ensure dimensions match
        query_mask = batch['query_mask'].squeeze(1)  # Remove channel dim to match pred_mask
        
        if idx == 0:
            print(f"Query mask shape after squeeze: {query_mask.shape}")
        
        # Calculate intersection and union directly
        pred_foreground = (pred_mask == 1)  # Assuming class 1 is the foreground
        gt_foreground = (query_mask > 0)    # Consider any positive value as foreground
        
        intersection = (pred_foreground & gt_foreground).sum().float()
        union = (pred_foreground | gt_foreground).sum().float()
        
        if idx == 0:
            print(f"Intersection: {intersection}, Union: {union}")
            print(f"Foreground pixels in pred: {pred_foreground.sum()}")
            print(f"Foreground pixels in gt: {gt_foreground.sum()}")
        
        total_inter += intersection
        total_union += union
        total_loss += loss.item() * batch['query_img'].size(0)
        samples_count += batch['query_img'].size(0)

        # Print progress
        if (idx+1) % 20 == 0:
            miou = total_inter / total_union if total_union > 0 else 0
            avg_loss = total_loss / samples_count
            duration = time.time() - start_time
            print(f"Epoch {epoch} | {phase} | Batch {idx+1}/{len(dataloader)} | Loss: {avg_loss:.4f} | mIoU: {miou:.4f} | Time: {duration:.2f}s")
    
    # Calculate final metrics
    miou = total_inter / total_union if total_union > 0 else 0
    avg_loss = total_loss / samples_count
    fb_iou = miou  # Use mIoU as fb_iou for simplicity

    print(f"{phase} Epoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {miou:.4f}")
    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    # Load configuration from fixed YAML path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../params/train.yaml'))
    print(f"Loading configuration from: {config_path}")
    args = load_config(config_path)

    # Create output directory for saving models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"./model_checkpoints/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved to: {save_dir}")

    # Model initialization
    model = DCAMA(args.backbone, args.feature_extractor_path, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Define paths to your 4 folders
    train_img_dir = os.path.join(args.datapath, 'train_images')
    train_mask_dir = os.path.join(args.datapath, 'train_masks')
    test_img_dir = os.path.join(args.datapath, 'test_images')
    test_mask_dir = os.path.join(args.datapath, 'test_masks')
    
    # Verify paths exist
    for path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist!")
    
    # Get dataloaders for custom dataset
    dataloader_trn, dataloader_val = get_custom_dataloader(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        test_img_dir=test_img_dir,
        test_mask_dir=test_mask_dir,
        batch_size=args.bsz,
        num_workers=args.nworker,
        n_shots=1,  # Number of support examples
        img_size=args.img_size,
        use_original_imgsize=args.use_original_imgsize
    )
    
    # Optimizer initialization
    optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
                          "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    Evaluator.initialize()

    # Training loop
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    
    for epoch in range(args.nepoch):
        print(f"\n{'='*20} Epoch {epoch+1}/{args.nepoch} {'='*20}")
        
        # Training phase
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)

        # Validation phase
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save based on mIoU
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), f"{save_dir}/best_miou_model.pt")
            print(f"Saved new best mIoU model: {best_val_miou:.4f}")
        
        # Save based on loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_loss_model.pt")
            print(f"Saved new best loss model: {best_val_loss:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.nepoch:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_miou': best_val_miou,
                'best_val_loss': best_val_loss
            }, f"{save_dir}/checkpoint_epoch{epoch+1}.pt")
            print(f"Saved checkpoint at epoch {epoch+1}")

    print('\n==================== Finished Training ====================')
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model checkpoints saved to: {save_dir}")