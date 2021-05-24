import os
from threading import Barrier
import time
import random
import argparse
import json
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util.warmup_scheduler import GradualWarmupSchedulerV2
from dataset import get_df, get_transforms, MMDataset
from models import Effnet, Resnest, Seresnext


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='./weights') # 不需要修改
    parser.add_argument('--log-dir', type=str, default='./logs') # 不需要修改
    parser.add_argument('--label-dir', type=str, default='./datasets') # 不需要修改
    parser.add_argument('--train-data-dir', type=str, default='./datasets/images/ISIC2020/jpeg/train_1024')
    parser.add_argument('--test-data-dir', type=str, default='./datasets/images/ISIC2020/jpeg/test_1024')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--enet-type', type=str, default='efficientnet_b3')
    parser.add_argument('--kernel-type', type=str) # 模型保存名字，不指定则使用默认名称，不需要修改
    parser.add_argument('--out-dim', type=int, default=9) # 9分类
    parser.add_argument('--image-size', type=int, default=512)  # resize后的图像大小
    parser.add_argument('--fold-type',type=str,default='') # 将20个fold映射为五个，可选为'fold+' 'fold++' ''
    parser.add_argument('--train-fold', type=str, default='0,1,2,3,4') # train folds分别作为验证集
    parser.add_argument('--freeze-cnn', action='store_true', default=False) # 冻结CNN参数
    parser.add_argument('--DANN', action='store_true', default=False) # 是否使用DANN毛发消除
    parser.add_argument('--use-meta', action='store_true', default=True) # 是否使用meta
    parser.add_argument('--meta-model', type=str, default='joint') # meta模型,joint or adadec
    parser.add_argument('--cc', action='store_true', default=True) # color constancy
    parser.add_argument('--cc-method', type=str, default='max_rgb') # color constancy method
    parser.add_argument('--n_meta_dim', type=str, default='512,128')
    parser.add_argument('--DEBUG', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=16)
    args, _ = parser.parse_known_args()
    return args


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, mel_idx, n_test=1, get_output=False):
    model.eval()
    class_val_loss = []
    barrier_val_loss = []
    CLASS_LOGITS = []
    BARRIER_LOGITS = []
    CLASS_PROBS = []
    BARRIER_PROBS = []
    CLASS_TARGETS = []
    BARRIER_TARGETS = []
    if args.DANN:
        with torch.no_grad():
            for (data, target) in tqdm(loader):
                if args.use_meta:
                    data, meta = data
                    target_class, target_barrier = target
                    data, meta, target_class, target_barrier = data.to(device), meta.to(device), target_class.to(device), target_barrier.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    barrier_logits = torch.zeros((data.shape[0], 2)).to(device)
                    barrier_probs = torch.zeros((data.shape[0], 2)).to(device)
                    for I in range(n_test):
                        l1,l2 = model(get_trans(data, I), meta)
                        class_logits += l1
                        barrier_logits += l2
                        class_probs += l1.softmax(1)
                        barrier_probs += l2.softmax(1)
                else:
                    target_class, target_barrier = target
                    data, target_class, target_barrier = data.to(device), target_class.to(device), target_barrier.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    barrier_logits = torch.zeros((data.shape[0], 2)).to(device)
                    barrier_probs = torch.zeros((data.shape[0], 2)).to(device)
                    for I in range(n_test):
                        l1,l2 = model(get_trans(data, I))
                        class_logits += l1
                        barrier_logits += l2
                        class_probs += l1.softmax(1)
                        barrier_probs += l2.softmax(1)      
                class_logits /= n_test
                barrier_logits /= n_test
                class_probs /= n_test
                barrier_probs /= n_test
                CLASS_LOGITS.append(class_logits.detach().cpu())
                CLASS_PROBS.append(class_probs.detach().cpu())
                CLASS_TARGETS.append(target_class.detach().cpu())
                BARRIER_LOGITS.append(barrier_logits.detach().cpu())
                BARRIER_PROBS.append(barrier_probs.detach().cpu())
                BARRIER_TARGETS.append(target_barrier.detach().cpu())
                class_loss = criterion(class_logits, target_class)
                class_val_loss.append(class_loss.detach().cpu().numpy())
                barrier_loss = barrier_criterion(barrier_logits, target_barrier)
                barrier_val_loss.append(barrier_loss.detach().cpu().numpy())
                
        class_val_loss = np.mean(class_val_loss)
        barrier_val_loss = np.mean(barrier_val_loss)

        CLASS_LOGITS = torch.cat(CLASS_LOGITS).numpy()
        CLASS_PROBS = torch.cat(CLASS_PROBS).numpy()
        CLASS_TARGETS = torch.cat(CLASS_TARGETS).numpy()
        BARRIER_LOGITS = torch.cat(BARRIER_LOGITS).numpy()
        BARRIER_PROBS = torch.cat(BARRIER_PROBS).numpy()
        BARRIER_TARGETS = torch.cat(BARRIER_TARGETS).numpy()

        if get_output:
            return CLASS_LOGITS, CLASS_PROBS, BARRIER_LOGITS, BARRIER_PROBS
        else:
            class_acc = (CLASS_PROBS.argmax(1) == CLASS_TARGETS).mean() * 100.
            class_auc = roc_auc_score((CLASS_TARGETS == mel_idx).astype(float), CLASS_PROBS[:, mel_idx])
            barrier_acc = (BARRIER_PROBS.argmax(1) == BARRIER_TARGETS).mean() * 100.
            barrier_auc = roc_auc_score((BARRIER_TARGETS == 0).astype(float), BARRIER_PROBS[:, 0])
            class_acc_list_0=[]
            class_acc_list_1=[]
            for i in range(len(CLASS_TARGETS)):
                if int(CLASS_TARGETS[i])!=mel_idx:
                    class_acc_list_0.append(CLASS_PROBS.argmax(1)[i]==CLASS_TARGETS[i])
                elif int(CLASS_TARGETS[i])==mel_idx:
                    class_acc_list_1.append(CLASS_PROBS.argmax(1)[i]==CLASS_TARGETS[i])
            class_acc_0 = np.array(class_acc_list_0).mean() * 100.
            class_acc_1 = np.array(class_acc_list_1).mean() * 100.
            
            barrier_acc_list_0=[]
            barrier_acc_list_1=[]
            for i in range(len(BARRIER_TARGETS)):
                if int(BARRIER_TARGETS[i])==0:
                    barrier_acc_list_0.append(BARRIER_PROBS.argmax(1)[i]==BARRIER_TARGETS[i])
                elif int(BARRIER_TARGETS[i])==1:
                    barrier_acc_list_1.append(BARRIER_PROBS.argmax(1)[i]==BARRIER_TARGETS[i])
            barrier_acc_0 = np.array(barrier_acc_list_0).mean() * 100.
            barrier_acc_1 = np.array(barrier_acc_list_1).mean() * 100.
            return class_val_loss, class_acc, class_auc, class_acc_0, class_acc_1, barrier_val_loss, barrier_acc, barrier_auc, barrier_acc_0, barrier_acc_1
    else:
        with torch.no_grad():
            for (data, target) in tqdm(loader):
                if args.use_meta:
                    data, meta = data
                    target_class = target
                    data, meta, target_class = data.to(device), meta.to(device), target_class.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(n_test):
                        l1 = model(get_trans(data, I), meta)
                        class_logits += l1
                        class_probs += l1.softmax(1)
                else:
                    target_class = target
                    data, target_class = data.to(device), target_class.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(n_test):
                        l1 = model(get_trans(data, I))
                        class_logits += l1
                        class_probs += l1.softmax(1)
                class_logits /= n_test
                class_probs /= n_test
                CLASS_LOGITS.append(class_logits.detach().cpu())
                CLASS_PROBS.append(class_probs.detach().cpu())
                CLASS_TARGETS.append(target_class.detach().cpu())
                class_loss = criterion(class_logits, target_class)
                class_val_loss.append(class_loss.detach().cpu().numpy())
                
        class_val_loss = np.mean(class_val_loss)

        CLASS_LOGITS = torch.cat(CLASS_LOGITS).numpy()
        CLASS_PROBS = torch.cat(CLASS_PROBS).numpy()
        CLASS_TARGETS = torch.cat(CLASS_TARGETS).numpy()

        if get_output:
            return CLASS_LOGITS, CLASS_PROBS
        else:
            class_acc = (CLASS_PROBS.argmax(1) == CLASS_TARGETS).mean() * 100.
            class_auc = roc_auc_score((CLASS_TARGETS == mel_idx).astype(float), CLASS_PROBS[:, mel_idx])
            class_acc_list_0=[]
            class_acc_list_1=[]
            for i in range(len(CLASS_TARGETS)):
                if int(CLASS_TARGETS[i])!=mel_idx:
                    class_acc_list_0.append(CLASS_PROBS.argmax(1)[i]==CLASS_TARGETS[i])
                elif int(CLASS_TARGETS[i])==mel_idx:
                    class_acc_list_1.append(CLASS_PROBS.argmax(1)[i]==CLASS_TARGETS[i])
                    
            class_acc_0 = np.array(class_acc_list_0).mean() * 100.
            class_acc_1 = np.array(class_acc_list_1).mean() * 100.
            
            return class_val_loss, class_acc, class_auc, class_acc_0, class_acc_1

def train_epoch(epoch, model, loader, optimizer):
    model.train()
    train_loss = []
    barrier_train_loss = []
    bar = tqdm(loader)
    i=0
    for (data, target) in bar:  # [tensor(batchsize, 3, 512, 512),tensor(batchsize, 10)]  tensor(2)
        p = float(i + (epoch-1) * len(bar)) / args.n_epochs / len(bar)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        i+=1
        optimizer.zero_grad()
        
        if args.DANN:
            if args.use_meta:
                data, meta = data
                target_class, target_barrier = target
                data, meta, target_class, target_barrier = data.to(device), meta.to(device), target_class.to(device), target_barrier.to(device)
                class_out,barrier_out = model(data, x_meta=meta, alpha=alpha)
            else:
                target_class, target_barrier = target
                data, target_class, target_barrier = data.to(device), target_class.to(device), target_barrier.to(device)
                class_out,barrier_out = model(data, alpha=alpha)        
            class_loss = criterion(class_out, target_class)
            barrier_loss = barrier_criterion(barrier_out, target_barrier)
            loss = class_loss + barrier_loss
        else:
            if args.use_meta:
                data, meta = data
                target_class = target
                data, meta, target_class = data.to(device), meta.to(device), target_class.to(device)
                class_out = model(data, meta, alpha=alpha)
            else:
                target_class = target
                data, target_class = data.to(device), target_class.to(device)
                class_out = model(data, alpha=alpha)        
            class_loss = criterion(class_out, target_class)
            loss = class_loss

        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        if args.DANN:
            barrier_train_loss.append(barrier_loss.detach().cpu().numpy())
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    if args.DANN:
        barrier_train_loss = np.mean(barrier_train_loss)
        return train_loss, barrier_train_loss
    else:
        return train_loss, 0


def run(fold, df, transforms_train, transforms_val, _idx, log_file):
    with open(log_file, 'w') as appender:
        args_str = json.dumps(vars(args), indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':'))
        appender.write(args_str+"\n")
    with open(log_file, 'w') as appender:
        args_str = json.dumps(vars(args), indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':'))
        appender.write(args_str+"\n")
        
    if args.DEBUG:
        args.n_epochs = 3
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

    dataset_train = MMDataset(args, df_train, 'train', transform=transforms_train)
    dataset_valid = MMDataset(args, df_valid, 'valid', transform=transforms_val)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               sampler=RandomSampler(dataset_train),
                                               num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)
    model = ModelClass(args)
    if DP:
        pass
    model = model.to(device)

    auc_max = 0.
    
    model_file_best  = os.path.join(args.model_dir, f'{args.kernel_type}_best.pth')
    model_file_final = os.path.join(args.model_dir, f'{args.kernel_type}_final.pth')

    if args.freeze_cnn:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if DP:
        pass        
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)
    
    num_normal, num_mm, num_hair, num_unhair = dataset_train.get_num()
    content=f'total num of train:{len(dataset_train)},mm:{num_mm},normal:{num_normal},hair:{num_hair},unhair:{num_unhair}'+'\n'
    num_normal, num_mm, num_hair, num_unhair = dataset_valid.get_num()
    content+=f'total num of val:{len(dataset_valid)},mm:{num_mm},normal:{num_normal},hair:{num_hair},unhair:{num_unhair}'+'\n'
    with open(log_file, 'a') as appender:
        appender.write(content)

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}')

        train_loss, barrier_train_loss = train_epoch(epoch, model, train_loader, optimizer)
        if args.DANN:
            val_loss, acc, auc, acc_0, acc_1, barrier_val_loss, barrier_acc, barrier_auc, barrier_acc_0, barrier_acc_1= val_epoch(model, valid_loader, _idx)
            content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f} acc_0: {(acc_0):.6f}, acc_1: {(acc_1):.6f}.'+'\n'
            content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, barrier train loss: {barrier_train_loss:.5f}, barrier valid loss: {(barrier_val_loss):.5f}, acc: {(barrier_acc):.4f}, auc: {(barrier_auc):.6f} acc_0: {(barrier_acc_0):.6f}, acc_1: {(barrier_acc_1):.6f}.'
            print(content)
        else:
            val_loss, acc, auc, acc_0, acc_1= val_epoch(model, valid_loader, _idx)
            content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f} acc_0: {(acc_0):.6f}, acc_1: {(acc_1):.6f}.'
            print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')
        scheduler_warmup.step()
        if epoch == 2: scheduler_warmup.step()  # bug workaround

        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file_best)
            auc_max = auc

    torch.save(model.state_dict(), model_file_final)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    args = get_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir+'/debug', exist_ok=True)
    os.makedirs(args.log_dir+'/debug', exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed()

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()
    if args.DANN:
        barrier_criterion = nn.CrossEntropyLoss()
    
    df_train, df_test, _idx = get_df(args)
    transforms_train, transforms_val = get_transforms(args.image_size)
    folds = [int(i) for i in args.train_fold.split(',')]
    
    for fold in folds:
        if not args.kernel_type:
            args.kernel_type=f'{args.enet_type}_size{args.image_size}_outdim{args.out_dim}_bs{args.batch_size}'
            if args.freeze_cnn:
                args.kernel_type+='_freeze'
            if args.use_meta:
                args.kernel_type+='_meta'
                if args.meta_model=='joint':
                    args.kernel_type+='j' # metaj
                if args.meta_model=='adadec':
                    args.kernel_type+='a' # metaa  
            if args.DANN:
                args.kernel_type+='_DANN'
        args.kernel_type+=f'_{args.fold_type}{fold}'
        if args.DEBUG:
            log_file=os.path.join(args.log_dir+'/debug', f'log_{args.kernel_type}.txt')
        else:
            log_file=os.path.join(args.log_dir, f'log_{args.kernel_type}.txt')
        run(fold, df_train, transforms_train, transforms_val, _idx, log_file)
    