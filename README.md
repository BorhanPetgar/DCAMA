# Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation

This is a customized fork of the official implementation of the ECCV'2022 paper "Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation".

## Requirements

- Python 3.7+
- PyTorch 1.5.1+
- CUDA 10.1+
- timm
- tensorboard 1.14
- PyYAML
- Pillow
- NumPy

You can install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Alternatively, create a conda environment:

```bash
conda create -n DCAMA python=3.7
conda activate DCAMA

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX timm pillow pyyaml numpy
```

## Project Structure

```
DCAMA/                          # Project root
├── common/                     # Helper functions
├── data/                       # Dataloaders and dataset tools
├── model/                      # Model implementation
│   ├── base/                   # Base models (Transformer, Swin)
│   └── DCAMA.py                # Main DCAMA model
├── src/                        # Scripts for running the model
│   ├── data_preparation/       # Tools for preparing datasets
│   │   ├── add_mask_suffix.py  # Add suffix to mask files
│   │   ├── clahe.py            # Contrast enhancement
│   │   ├── download.py         # Download datasets
│   │   └── mask_gen.py         # Generate masks
│   ├── inference/              # Inference code
│   │   └── inference.py        # Run inference on images
│   └── train/                  # Training code
│       └── custom_train2.py    # Train on custom datasets
├── params/                     # Configuration files
│   ├── inference.yaml          # Inference configuration
│   └── train.yaml              # Training configuration
├── backbones/                  # Directory for pretrained backbones
├── model_checkpoints/          # Directory for saved model checkpoints
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

## Prepare Datasets

The model expects datasets in a specific format with the following directory structure:

```
dataset_folder/
├── train_images/     # Training images
├── train_masks/      # Binary masks for training images 
├── test_images/      # Test images
└── test_masks/       # Binary masks for test images
```

For support and query examples in inference, image-mask pairs should follow this naming convention:
- Images: name.jpg/png/bmp
- Masks: name_mask.png

You can use the script in add_mask_suffix.py to add the proper suffix to your mask files.

## Prepare Backbones

NOTE: The Code was tested on Swin-B

Download the following pre-trained backbones:

> 1. [ResNet-50](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h-35c100f8.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)
> 2. [ResNet-101](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)
> 3. [Swin-B](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth) pretrained on ImageNet by [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

Create a directory 'backbones' to place the above backbones.

## Model Checkpoints

During training, model checkpoints will be saved to the model_checkpoints directory with a timestamp for each training run (e.g., `model_checkpoints/20250408_120000/`). Each checkpoint directory contains:

- `best_miou_model.pt`: Model weights with the best mean IoU on validation set
- `latest_model.pt`: Most recently saved model weights
- `training_log.txt`: Training progress log
- `config.yaml`: Configuration used for this training run

When performing inference, specify the path to the desired checkpoint in the `load` parameter of your configuration file.

## Training

The model supports training on custom datasets. Use the following command to train:

```bash
python src/train/custom_train2.py \
       --datapath /path/to/your/dataset \
       --backbone swin \
       --feature_extractor_path backbones/swin_base_patch4_window12_384_22kto1k.pth \
       --nepoch 50 \
       --lr 0.001 \
       --bsz 4 \
       --nworker 1
```

Configuration options can be modified in train.yaml.

## Inference

For inference, you need to prepare:
1. Support images and masks (examples with annotations)
2. Query images (new images to segment)

Run inference with:

```bash
python src/inference/inference.py
```

This will use the configuration specified in inference.yaml. Key parameters include:

- `backbone`: Model backbone (swin, resnet50, resnet101)
- `feature_extractor_path`: Path to pretrained backbone
- `load`: Path to trained model checkpoint
- `support_folder`: Folder containing support images and their masks
- `query_img`: Path to query image
- `output_mask`: Path to save output mask
- `use_original_imgsize`: Whether to output at original image size
- `img_size`: Input size for model (must be 384 for Swin Transformer)

**Important Note:** When using the Swin Transformer backbone, the input image size must be 384x384. Set `img_size: 384` in the configuration. The `use_original_imgsize` parameter controls whether the output is resized to match the original image dimensions.

## Configuration Files

### inference.yaml
Contains parameters for inference:
```yaml
backbone: "swin"
feature_extractor_path: "backbones/swin_base_patch4_window12_384_22kto1k.pth"
load: "model_checkpoints/20250327_120422/best_miou_model.pt"
support_folder: "path/to/support/images"
query_img: "path/to/query/image"
output_mask: "path/to/output"
use_original_imgsize: True
img_size: 384
```

### train.yaml
Contains parameters for training (learning rate, batch size, epochs, etc.)

## requirements.txt

The repository includes a `requirements.txt` file with the following dependencies:

```
torch==1.5.1
torchvision==0.6.1
timm==0.5.4
tensorboardX==2.5.1
PyYAML==6.0
Pillow==9.3.0
numpy==1.21.6
scikit-image==0.19.3
opencv-python==4.6.0.66
```

Install these dependencies using:

```bash
pip install -r requirements.txt
```

## Data Preparation Utilities

The data_preparation folder contains utilities for preparing datasets:

- `add_mask_suffix.py`: Adds "_mask" suffix to mask files
- `clahe.py`: Applies contrast enhancement to images
- `download.py`: Utility for downloading datasets
- `mask_gen.py`: Tools for generating masks

## Asphalt Crack Dataset

For experiments on crack detection, the DeepCrack dataset can be used:
https://github.com/yhlleo/DeepCrack

Similar code found with 1 license type