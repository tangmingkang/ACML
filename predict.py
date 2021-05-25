import os
import argparse
from tqdm import tqdm
from dataset import get_df, get_transforms, MMDataset
from models import Effnet, Resnest, Seresnext
from train import get_trans
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='./weights') # 不需要修改
    parser.add_argument('--log-dir', type=str, default='./logs') # 不需要修改
    parser.add_argument('--label-dir', type=str, default='./datasets') # 不需要修改
    parser.add_argument('--train-data-dir', type=str, default='./datasets/images/ISIC2020/jpeg/train_1024')
    parser.add_argument('--test-data-dir', type=str, default='./datasets/images/ISIC2020/jpeg/test_1024')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='6')
    parser.add_argument('--enet-type', type=str, default='efficientnet_b3')
    parser.add_argument('--kernel-type', type=str, default='efficientnet_b3_size512_outdim9_bs64_metaj') # 指定需要测试的模型
    parser.add_argument('--out-dim', type=int, default=9) # 9分类
    parser.add_argument('--image-size', type=int, default=512)  # resize后的图像大小
    parser.add_argument('--val-fold', type=str, default='0') # val folds 模型平均值作为最终结果
    parser.add_argument('--DANN', action='store_true', default=False) # 是否使用DANN毛发消除
    parser.add_argument('--use-meta', action='store_true', default=True) # 是否使用meta
    parser.add_argument('--meta-model', type=str, default='joint') # meta模型,joint or adadec
    parser.add_argument('--cc', action='store_true', default=True) # color constancy
    parser.add_argument('--cc-method', type=str, default='max_rgb') # color constancy method
    parser.add_argument('--n_meta_dim', type=str, default='512,128')
    parser.add_argument('--DEBUG', action='store_true', default=False)
    parser.add_argument('--batch-val-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--eval', type=str, choices=['best', 'final'], default="best")
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--fold-type',type=str,default='') # 将20个fold映射为五个，可选为'fold+' 'fold++' ''
    args, _ = parser.parse_known_args()
    return args


def run(df_test, transforms_val):
    if args.DEBUG:
        df_test = df_test.sample(args.batch_val_size * 3)
    dataset_test = MMDataset(args, df_test, 'test', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_val_size, num_workers=args.num_workers)
    folds = [int(i) for i in args.val_fold.split(',')]
    models = []
    kernel_type=args.kernel_type
    for fold in folds:
        args.kernel_type=f'{kernel_type}_{args.fold_type}{fold}'
        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best.pth')
        elif args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final.pth')
        model = ModelClass(args)
        model = model.to(device)
        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        # if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        #     model = torch.nn.DataParallel(model)

        model.eval()
        models.append(model)

    # predict
    PROBS = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            if args.use_meta:
                data, meta = data
                data, meta = data.to(device), meta.to(device)
            else:
                data = data.to(device)        
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            for model in models:
                for I in range(args.n_test):
                    if args.use_meta:
                        l = model(get_trans(data, I),meta,test=True)
                    else:
                        l = model(get_trans(data, I),test=True)
                    probs += l.softmax(1)
            probs /= args.n_test
            probs /= len(models)
            PROBS.append(probs.detach().cpu())

    PROBS = torch.cat(PROBS).numpy()
    df_test['target'] = PROBS[:, _idx]
    df_test[['image_name', 'target']].to_csv(os.path.join(args.sub_dir, f'sub_{kernel_type}_{args.val_fold}_{args.eval}.csv'), index=False)

if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    if args.enet_type == 'resnest101':
        ModelClass = Resnest
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet
    else:
        raise NotImplementedError()
    
    device = torch.device('cuda')
    df_train, df_test, _idx = get_df(args)
    transforms_train, transforms_val = get_transforms(args.image_size)
    run(df_test, transforms_val)