import time
import os
folds=[0,1,2,3,4]
# 组件：cc meta DANN
train_name='train_efficientb5_cc_metaj_DANN2_focalloss'
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
    time.sleep(30)
    f=open(lock_name,'w')
    f.write('1')
    f.close()
    command=f'nohup python -m torch.distributed.launch --nproc_per_node=4 train.py --train-fold {fold} --lock-file {lock_name}  --loss focal --CUDA_VISIBLE_DEVICES 0,1,2,3 >{log_name} 2>&1 &'
    print(command)
    os.system(command)