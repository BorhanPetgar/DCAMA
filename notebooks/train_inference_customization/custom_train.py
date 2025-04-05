r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset

import numpy as np
########### Inferece Starts ########### @Borhan
def inference(model, query_img, support_imgs, support_masks):
    """
    Run inference on a query image using support images and masks.
    
    Args:
        model: The DCAMA model
        query_img: A tensor of shape [1, 3, H, W] - The image to segment
        support_imgs: A tensor of shape [1, shot, 3, H, W] - Support images containing target class
        support_masks: A tensor of shape [1, shot, H, W] - Binary masks of the target class in support images
        
    Returns:
        pred_mask: A binary segmentation mask for the query image
    """
    model.eval()
    
    with torch.no_grad():
        # Move inputs to GPU
        query_img = query_img.cuda()
        support_imgs = support_imgs.cuda()
        support_masks = support_masks.cuda()
        
        # Forward pass
        logit_mask = model(query_img, support_imgs.squeeze(1), support_masks.squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)
        
    return pred_mask


def prepare_input_for_inference(query_path, support_paths, support_mask_paths, transform=None):
    """
    Prepare input images and masks for inference.
    
    Args:
        query_path: Path to the query image
        support_paths: List of paths to support images
        support_mask_paths: List of paths to support mask images
        transform: Optional transform to apply to images
        
    Returns:
        query_img, support_imgs, support_masks: Tensors ready for model inference
    """
    from PIL import Image
    import torchvision.transforms.functional as TF
    
    # Use the same transform as training if not specified
    if transform is None:
        transform = FSSDataset.transform
    
    # Load and process query image
    query_img = Image.open(query_path).convert('RGB')
    query_img = transform(query_img).unsqueeze(0)  # [1, 3, H, W]
    
    # Load and process support images and masks
    support_imgs = []
    support_masks = []
    
    for img_path, mask_path in zip(support_paths, support_mask_paths):
        # Process support image
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        support_imgs.append(img)
        
        # Process mask (binary)
        mask = Image.open(mask_path).convert('L')
        mask = TF.resize(mask, (query_img.shape[2], query_img.shape[3]))
        mask = torch.from_numpy(np.array(mask) > 0).float()
        support_masks.append(mask)
    
    # Stack support images and masks
    support_imgs = torch.stack(support_imgs).unsqueeze(0)  # [1, shot, 3, H, W]
    support_masks = torch.stack(support_masks).unsqueeze(0)  # [1, shot, H, W]
    
    return query_img, support_imgs, support_masks

"""
# Example usage for different shot settings
import numpy as np
from PIL import Image

# Add this at the end of your script or in a separate file
if __name__ == '__main__':
    # Load a pretrained model
    model = DCAMA(backbone='resnet50', feature_extractor_path='path/to/weights', use_original_imgsize=False)
    model.load_state_dict(torch.load('path/to/saved_model.pth'))
    model.cuda()
    model.eval()
    
    # 1-shot example
    query_path = 'path/to/query.jpg'
    support_path = ['path/to/support.jpg']
    mask_path = ['path/to/support_mask.png']
    
    query_img, support_imgs, support_masks = prepare_input_for_inference(
        query_path, support_path, mask_path
    )
    
    # Run inference
    pred_mask = inference(model, query_img, support_imgs, support_masks)
    
    # Convert to numpy and save
    pred_mask = pred_mask[0].cpu().numpy()
    
    # Visualization
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    mask_img.save('output_mask.png')
    
    # For 5-shot, just provide 5 support images and masks
    support_path_5shot = ['support1.jpg', 'support2.jpg', 'support3.jpg', 'support4.jpg', 'support5.jpg']
    mask_path_5shot = ['mask1.png', 'mask2.png', 'mask3.png', 'mask4.png', 'mask5.png']
"""

########### Inferece Ends ########### @Borhan

def train(epoch, model, dataloader, optimizer, training):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


# if __name__ == '__main__':

#     # Arguments parsing
#     args = parse_opts()
#     args.local_rank = 0

#     # ddp backend initialization
#     # torch.distributed.init_process_group(backend='nccl')
#     # args.local_rank = torch.distributed.get_rank()
#     # torch.cuda.set_device(args.local_rank)

#     # Model initialization
#     model = DCAMA(args.backbone, args.feature_extractor_path, False)
#     device = torch.device("cuda", 0)
#     model.to(device)
#     # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
#     #                                             find_unused_parameters=True)

#     # Helper classes (for training) initialization
#     optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
#                             "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
#     Evaluator.initialize()
#     if args.local_rank == 0:
#         Logger.initialize(args, training=True)
#         Logger.info('# available GPUs: %d' % torch.cuda.device_count())

#     # Dataset initialization
#     FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
#     dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
#     if args.local_rank == 0:
#         dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

#     # Train
#     best_val_miou = float('-inf')
#     best_val_loss = float('inf')
#     for epoch in range(args.nepoch):
#         # dataloader_trn.sampler.set_epoch(epoch)
#         trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)

#         # evaluation
#         if args.local_rank == 0:
#             with torch.no_grad():
#                 val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

#             # Save the best model
#             if val_miou > best_val_miou:
#                 best_val_miou = val_miou
#                 Logger.save_model_miou(model, epoch, val_miou)

#             Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
#             Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
#             Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
#             Logger.tbd_writer.flush()

#     if args.local_rank == 0:
#         Logger.tbd_writer.close()
#         Logger.info('==================== Finished Training ====================')

import numpy as np
from PIL import Image

# Add this at the end of your script or in a separate file
if __name__ == '__main__':
    # Load a pretrained model
    model = DCAMA(backbone='swin', pretrained_path='/home/ubuntu/borhan/projects/anomaly/repos/DCAMA/backbones/swin_base_patch4_window12_384_22kto1k.pth', use_original_imgsize=False)

    model.load_state_dict(torch.load('/home/ubuntu/borhan/projects/anomaly/repos/DCAMA/checkpoints/swin_fold1.pt'))
    model.cuda()
    model.eval()
    
    # 1-shot example
    query_path = '/home/ubuntu/borhan/projects/anomaly/repos/DCAMA/my_data/q1.jpg'
    support_path = ['/home/ubuntu/borhan/projects/anomaly/repos/DCAMA/my_data/s1.jpg']
    mask_path = ['/home/ubuntu/borhan/projects/anomaly/repos/DCAMA/my_data/s1_mask.png']
    
    query_img, support_imgs, support_masks = prepare_input_for_inference(
        query_path, support_path, mask_path
    )
    
    # Run inference
    pred_mask = inference(model, query_img, support_imgs, support_masks)
    
    # Convert to numpy and save
    pred_mask = pred_mask[0].cpu().numpy()
    
    # Visualization
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    mask_img.save('output_mask.png')
    
    # For 5-shot, just provide 5 support images and masks
    # support_path_5shot = ['support1.jpg', 'support2.jpg', 'support3.jpg', 'support4.jpg', 'support5.jpg']
    # mask_path_5shot = ['mask1.png', 'mask2.png', 'mask3.png', 'mask4.png', 'mask5.png']