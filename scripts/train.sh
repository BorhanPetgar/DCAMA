# python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=16005 \
python3 \
./train.py --datapath "/home/ubuntu/borhan/projects/anomaly/repos/DCAMA/datasets" \
           --benchmark coco \
           --fold 0 \
           --bsz 12 \
           --nworker 8 \
           --backbone swin \
           --feature_extractor_path "/home/ubuntu/borhan/projects/anomaly/repos/DCAMA/backbones/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "./logs" \
           --lr 1e-3 \
           --nepoch 500


# /home/ubuntu/borhan/projects/anomaly/repos/DCAMA/backbones/swin_base_patch4_window12_384_22kto1k.pth