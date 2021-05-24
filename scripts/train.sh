log_name='train.log'
nohup python train.py \
    --CUDA_VISIBLE_DEVICES 0 \
    --train-data-dir ./datasets/images/ISIC2020/jpeg/train_1024 `# 图像路径，修改为自己的路径` \
    --test-data-dir ./datasets/images/ISIC2020/jpeg/train_1024 `# 图像路径，修改为自己的路径` \
    --train-fold 0,1,2,3,4 `# 五折交叉验证 可只选择其中几个做验证` \
    --DEBUG `# DEBUG模式epoch=3，随机选择4个batchsize的数据作为数据集` \
    --use-meta `# 使用meta数据` \
    --cc `# 色彩恒常` \
    --batch-size 16 \
    --init-lr 3e-5 \
    --n-epochs 15 \
    --num-workers 16 \
>$log_name 2>&1 &