import time
import os
folds=[0,1,2,3,4]
for fold in folds:
    log_name=f'train{fold}.log'
    while True:
        f=open('lock.txt','r')
        if f.readline()=='0': 
            f.close()
            break
        else: time.sleep(60)
    f=open('lock.txt','w')
    f.write('1')
    f.close()
    command=f'nohup python -m torch.distributed.launch --nproc_per_node=4 train.py --train-fold {fold} >{log_name} 2>&1 &'
    print(command)
    os.system(command)