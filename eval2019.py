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
import sklearn
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from util.focal_loss import FocalLoss
from util.warmup_scheduler import GradualWarmupSchedulerV2
from dataset2 import get_df, get_transforms, MMDataset
from models import Effnet, Resnest, Seresnext

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='./weights') # 不需要修改
    parser.add_argument('--log-dir', type=str, default='./logs') # 不需要修改
    parser.add_argument('--label-dir', type=str, default='./datasets') # 不需要修改
    parser.add_argument('--train-data-dir', type=str, default='./datasets/images/ISIC2020/jpeg/train_1024')
    parser.add_argument('--test-data-dir', type=str, default='./datasets/images/ISIC2020/jpeg/test_1024')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='6')
    parser.add_argument('--enet-type', type=str, default='efficientnet_b3')
    parser.add_argument('--kernel-type', type=str, default='efficientnet_b3_size512_outdim9_bs32_celoss_ccmax_rgb_DANN2_fakedata') # 指定用于验证的模型
    parser.add_argument('--out-dim', type=int, default=9) # 9分类
    parser.add_argument('--image-size', type=int, default=512)  # resize后的图像大小
    parser.add_argument('--fold-type',type=str,default='') # 将20个fold映射为五个，可选为'fold+' 'fold++' ''
    parser.add_argument('--val-fold', type=str, default='2') # val folds分别作为验证集
    parser.add_argument('--DANN', action='store_true', default=True) # 是否使用DANN毛发消除
    parser.add_argument('--n-dann-dim', default=2) # DANN class num
    parser.add_argument('--use-meta', action='store_true', default=False) # 是否使用meta
    parser.add_argument('--meta-model', type=str, default='joint') # meta模型,joint or adadec
    parser.add_argument('--cc', action='store_true', default=True) # color constancy
    parser.add_argument('--cc-method', type=str, default='max_rgb') # color constancy method
    parser.add_argument('--remove', action='store_true', default=False) # color constancy
    parser.add_argument('--remove-method', type=str, default='rm1') # color constancy method
    parser.add_argument('--n_meta_dim', type=str, default='512,128')
    parser.add_argument('--DEBUG', action='store_true', default=False)
    parser.add_argument('--eval', type=str, choices=['best', 'final'], default="auc")
    parser.add_argument('--batch-eval-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce','focal','wce']) # ce,focal,wce
    parser.add_argument('--wcew', type=float, default=25) # ce,focal,wce
    parser.add_argument('--fake', action='store_true', default=False) # color constancy

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

def get_metrics(y, score):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, score)
    auc = sklearn.metrics.auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    
    optimal_th = thresholds[optimal_idx]
    
    final_th=thresholds[int(len(thresholds)*0.74)]
    
    print(optimal_th)
    print(final_th)
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    predict_labels = [1 if _score >= final_th else 0 for _score in score]
    for idx in range(len(predict_labels)):
        predict_label = predict_labels[idx]
        ture_label = y[idx]
        if ture_label == 1 and predict_label == 1:
            tp += 1
        if ture_label == 1 and predict_label == 0:
            fn += 1
        if ture_label == 0 and predict_label == 0:
            tn += 1
        if ture_label == 0 and predict_label == 1:
            fp += 1
    
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = tp/(tp+fn)
    spec = tn/(tn+fp)
    if tp+fp==0:
        pre=1.0
    else:   
        pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1=2*pre*rec/(pre+rec)
    return auc,acc,sen,spec,pre,rec,f1

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
                    barrier_logits = torch.zeros((data.shape[0], args.n_dann_dim)).to(device)
                    barrier_probs = torch.zeros((data.shape[0], args.n_dann_dim)).to(device)
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
                    barrier_logits = torch.zeros((data.shape[0], args.n_dann_dim)).to(device)
                    barrier_probs = torch.zeros((data.shape[0], args.n_dann_dim)).to(device)
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
                class_loss = criterion(class_logits, target_class)
                class_val_loss.append(class_loss.detach().cpu().numpy())
                barrier_loss = barrier_criterion(barrier_logits, target_barrier)
                barrier_val_loss.append(barrier_loss.detach().cpu().numpy())
                class_probs=class_probs.detach().cpu()
                class_probs=[[1.-prob[mel_idx],prob[mel_idx]] for prob in class_probs]
                class_probs=torch.Tensor(class_probs)
                target_class=target_class.detach().cpu()
                target_class=[ 0 if target!=mel_idx else 1 for target in target_class]
                target_class=torch.IntTensor(target_class)
                CLASS_LOGITS.append(class_logits.detach().cpu())
                CLASS_PROBS.append(class_probs)
                CLASS_TARGETS.append(target_class)
                BARRIER_LOGITS.append(barrier_logits.detach().cpu())
                BARRIER_PROBS.append(barrier_probs.detach().cpu())
                BARRIER_TARGETS.append(target_barrier.detach().cpu())
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
            class_auc,class_acc,class_sen,class_spec,class_pre,class_rec,class_f1=get_metrics((CLASS_TARGETS == 1).astype(float), CLASS_PROBS[:, 1]) # 由于上面对取值做了修改  无论9还是2分类这里都是1
            barrier_auc,barrier_acc,barrier_sen,barrier_spec,barrier_pre,barrier_rec,barrier_f1=get_metrics((CLASS_TARGETS == 1).astype(float), CLASS_PROBS[:, 1])
            class_acc_0=class_spec
            class_acc_1=class_sen
            barrier_acc_0=barrier_spec
            barrier_acc_1=barrier_sen
            return class_val_loss, class_acc, class_auc, class_acc_0, class_acc_1, class_pre, class_rec, class_f1, class_sen, class_spec, barrier_val_loss, barrier_acc, barrier_auc, barrier_acc_0, barrier_acc_1
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
                class_loss = criterion(class_logits, target_class)
                class_val_loss.append(class_loss.detach().cpu().numpy())
                class_probs=class_probs.detach().cpu()
                class_probs=[[1.-prob[mel_idx],prob[mel_idx]] for prob in class_probs]
                class_probs=torch.Tensor(class_probs)
                target_class=target_class.detach().cpu()
                target_class=[ 0 if target!=mel_idx else 1 for target in target_class]
                target_class=torch.IntTensor(target_class)
                CLASS_LOGITS.append(class_logits.detach().cpu())
                CLASS_PROBS.append(class_probs)
                CLASS_TARGETS.append(target_class)
                
        class_val_loss = np.mean(class_val_loss)

        CLASS_LOGITS = torch.cat(CLASS_LOGITS).numpy()
        CLASS_PROBS = torch.cat(CLASS_PROBS).numpy()
        CLASS_TARGETS = torch.cat(CLASS_TARGETS).numpy()

        if get_output:
            return CLASS_LOGITS, CLASS_PROBS
        else:
            class_auc,class_acc,class_sen,class_spec,class_pre,class_rec,class_f1=get_metrics((CLASS_TARGETS == 1).astype(float), CLASS_PROBS[:, 1])
            class_acc_0=class_spec
            class_acc_1=class_sen
            return class_val_loss, class_acc, class_auc, class_acc_0, class_acc_1, class_pre, class_rec, class_f1, class_sen, class_spec

def train_epoch(epoch, model, loader, optimizer):
    model.train()
    train_loss = []
    barrier_train_loss = []
    if args.local_rank==0:
        bar = tqdm(loader)
    else:
        bar = loader
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
        if args.local_rank==0:
            bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    if args.DANN:
        barrier_train_loss = np.mean(barrier_train_loss)
        return train_loss, barrier_train_loss
    else:
        return train_loss, 0


def run(fold, df, transforms_val, _idx, log_file):
    with open(log_file, 'a') as appender:
        appender.write(f'fold{fold} {args.kernel_type}\n')
    
    # df_valid1 = df[df['target']==1].sample(200)
    # df_valid0 = df[df['target']==0].sample(1000)
    # df_valid = pd.concat([df_valid1, df_valid0]).reset_index(drop=True)
    df_valid=df
    dataset_valid = MMDataset(args, df_valid, 'valid', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_eval_size, num_workers=args.num_workers)
    model = ModelClass(args).to(device)
    # Find total parameters and trainable parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # exit()
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_{args.eval}.pth')
    
    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    num_normal, num_mm, num_hair, num_unhair = dataset_valid.get_num()
    content=f'total num of val:{len(dataset_valid)},mm:{num_mm},normal:{num_normal},hair:{num_hair},unhair:{num_unhair}'+'\n'
    with open(log_file, 'a') as appender:
        appender.write(content)
    if args.DANN:
        val_loss, acc, auc, acc_0, acc_1, class_pre, class_rec, class_f1, class_sen, class_spec, barrier_val_loss, barrier_acc, barrier_auc, barrier_acc_0, barrier_acc_1= val_epoch(model, valid_loader, _idx)
        content = time.ctime() + f'valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}, rec: {(class_rec):.6f}, pre: {(class_pre):.6f}, f1: {(class_f1):.6f}, spec: {(class_spec):.6f}.'+'\n'
        content += time.ctime() + f'barrier valid loss: {(barrier_val_loss):.5f}, acc: {(barrier_acc):.4f}, auc: {(barrier_auc):.6f} acc_0: {(barrier_acc_0):.6f}, acc_1: {(barrier_acc_1):.6f}.'
        print(content)
    else:
        val_loss, acc, auc, acc_0, acc_1, class_pre, class_rec, class_f1, class_sen, class_spec= val_epoch(model, valid_loader, _idx)
        content = time.ctime() + f'valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}, rec: {(class_rec):.6f}, pre: {(class_pre):.6f}, f1: {(class_f1):.6f}, spec: {(class_spec):.6f}.'+'\n'
        print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    ACC.append(acc)
    AUC.append(auc)
    PRE.append(class_pre)
    REC.append(class_rec)
    F1.append(class_f1)
    SEN.append(class_sen)
    SPEC.append(class_spec)
    ACC_0.append(acc_0)
    ACC_1.append(acc_1)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    ACC=[]
    AUC=[]
    ACC_0=[]
    ACC_1=[]
    PRE=[]
    REC=[]
    F1=[]
    SEN=[]
    SPEC=[]
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet
    else:
        raise NotImplementedError()

    device = torch.device("cuda")
    set_seed()
    
    
    df_train, df_test, _idx = get_df(args)
    
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        weight_CE = torch.FloatTensor([10]*args.out_dim)
        weight_CE[_idx]=10
        weight_CE=weight_CE.to(device)
        criterion = FocalLoss(args.out_dim,alpha=weight_CE)
    elif args.loss == 'wce':
        weight_CE = torch.FloatTensor([1]*args.out_dim)
        weight_CE[_idx]=args.wcew
        weight_CE=weight_CE.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_CE).cuda()
    if args.DANN:
        barrier_criterion = nn.CrossEntropyLoss()

    transforms_train, transforms_val = get_transforms(args.image_size)
    folds = [int(i) for i in args.val_fold.split(',')]
    kernel_type=args.kernel_type
    if args.DEBUG:
        log_file=os.path.join(args.log_dir+'/debug', f'log_{kernel_type}_{args.eval}val.txt')
    else:
        log_file=os.path.join(args.log_dir, f'log_{kernel_type}_{args.eval}val_ISIC2019.txt')
        
    for fold in folds:
        args.kernel_type=kernel_type
        args.kernel_type+=f'_{args.fold_type}{fold}'
        run(fold, df_train, transforms_val, _idx, log_file)

    with open(log_file, 'a') as appender:
        appender.write(f'all folds {args.val_fold} {kernel_type}\n')
        content = f'acc: {(np.mean(ACC)):.4f}, auc: {(np.mean(AUC)):.6f}, SPEC: {(np.mean(SPEC)):.6f}, PRE: {(np.mean(PRE)):.6f}, REC: {(np.mean(REC)):.6f}, F1: {(np.mean(F1)):.6f}.\n'
        content += f'acc: {(np.std(ACC)):.4f}, auc: {(np.std(AUC)):.6f}, SPEC: {(np.std(SPEC)):.6f}, PRE: {(np.std(PRE)):.6f}, REC: {(np.std(REC)):.6f}, F1: {(np.std(F1)):.6f}.'
        appender.write(content + '\n')
