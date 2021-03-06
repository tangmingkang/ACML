import time
import os
folds=[0,1,2,3,4,'e']
# 组件：cc meta DANN
train_name='train_origin_cc_fake'
lock_name=f'{train_name}_lock.txt'
f=open(lock_name,'w')
f.write('0')
f.close()
for fold in folds:
    log_name=f'{train_name}_{fold}.log'
    while True:
        f=open(lock_name,'r')
        if f.readline()=='0': 
            f.close()
            break
        else: time.sleep(60)
    if fold!=folds[0]:
        time.sleep(90)
    # --master_port 88888
    f=open(lock_name,'w')
    f.write('1')
    f.close()
    if fold!='e':
        command=f'nohup python -m torch.distributed.launch --master_port 88887 --nproc_per_node=4 train.py --train-fold {fold} --lock-file {lock_name}  --loss ce --CUDA_VISIBLE_DEVICES 4,5,6,7 --cc --fake --DANN --enet-type efficientnet_b3 --image-size 512>{log_name} 2>&1 &'
    else:
        command=f'python eval.py'
    print(command)
    os.system(command)