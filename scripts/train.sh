log_name='train.log'
nohup python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --CUDA_VISIBLE_DEVICES 0,1,4,5 \
    --train-data-dir ./datasets/images/ISIC2020/jpeg/train_1024 `# 图像路径，修改为自己的路径` \
    --test-data-dir ./datasets/images/ISIC2020/jpeg/train_1024 `# 图像路径，修改为自己的路径` \
    --train-fold 0,1,2,3,4 `# 五折交叉验证 可只选择其中几个做验证` \
    --use-meta `# 使用meta数据` \
    --cc `# 色彩恒常` \
    --batch-size 64 \
    --init-lr 3e-5 \
    --n-epochs 15 \
    --num-workers 8 \
>$log_name 2>&1 &