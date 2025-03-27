python inference.py \
            --backbone swin \
            --feature_extractor_path "/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/backbones/swin_base_patch4_window12_384_22kto1k.pth" \
            --load "/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/checkpoints/swin_fold0.pt" \
            --support_img "/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/my_data/s1.jpg" \
            --support_mask "/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/my_data/s1_mask.png" \
            --query_img "/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/my_data/q1.jpg" \
            --output_mask "/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/results/masks/q1_output_mask_15.png" \
            --nshot 1

# sh ./scripts/inference.sh