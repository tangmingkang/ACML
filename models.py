import torch
import torch.nn as nn
import geffnet
from torch.autograd import Function
from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d

sigmoid = nn.Sigmoid()

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Effnet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(Effnet, self).__init__()
        self.enet = geffnet.create_model(args.enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.n_meta_dim = [int(i) for i in args.n_meta_dim .split(',')]
        self.meta_model=args.meta_model
        self.DANN=args.DANN
        self.use_meta=args.use_meta
        in_ch = self.enet.classifier.in_features
        if args.use_meta:
            self.meta = nn.Sequential(
                nn.Linear(10, self.n_meta_dim[0]),
                nn.BatchNorm1d(self.n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(self.n_meta_dim[0], self.n_meta_dim[1]),
                nn.BatchNorm1d(self.n_meta_dim[1]),
                Swish_Module(),
            )
            if self.meta_model=='joint':
                in_ch += self.n_meta_dim[1]
            if self.meta_model=='adadec':
                self.attention=nn.Linear(self.n_meta_dim[1],in_ch)
        if args.DANN:
            self.barrier_classifier = nn.Sequential(
                nn.Linear(self.enet.classifier.in_features,100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Linear(100, args.n_dann_dim)
            )    
        self.myfc = nn.Linear(in_ch, args.out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None, alpha=0, test=False):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.DANN and not test:
            barrier_x = ReverseLayerF.apply(x,alpha)
            barrier_out = self.barrier_classifier(barrier_x)
        if self.use_meta:
            x_meta = self.meta(x_meta)
            if self.meta_model=='joint':
                x = torch.cat((x, x_meta), dim=1)
            elif self.meta_model=='adadec':
                att = self.attention(x_meta) # bs*n_meta_dim[1] -> bs*in_ch
                lamb_att = torch.diag_embed(att) # bs*in_ch*in_ch
                x = x.unsqueeze(1) # bs,1,in_ch
                x = x.bmm(lamb_att).squeeze(1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        if self.DANN and not test:
            return out,barrier_out
        else:
            return out 


class B6_bsline(nn.Module):
    def __init__(self, args, pretrained=True):
        super(B6_bsline, self).__init__()
        self.enet = geffnet.create_model('tf_efficientnet_b6_ns', pretrained=pretrained)
        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Sequential(
            nn.Linear(in_ch, args.out_dim),
            
            nn.Linear(10, self.n_meta_dim[0]),
            nn.BatchNorm1d(self.n_meta_dim[0]),
            Swish_Module(),
            nn.Linear(self.n_meta_dim[0], self.n_meta_dim[1]),
            nn.BatchNorm1d(self.n_meta_dim[1]),
            Swish_Module(),
        )
        # self.myfc = nn.Linear(in_ch, args.out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None, alpha=0, test=False):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.DANN and not test:
            barrier_x = ReverseLayerF.apply(x,alpha)
            barrier_out = self.barrier_classifier(barrier_x)
        if self.use_meta:
            x_meta = self.meta(x_meta)
            if self.meta_model=='joint':
                x = torch.cat((x, x_meta), dim=1)
            elif self.meta_model=='adadec':
                att = self.attention(x_meta) # bs*n_meta_dim[1] -> bs*in_ch
                lamb_att = torch.diag_embed(att) # bs*in_ch*in_ch
                x = x.unsqueeze(1) # bs,1,in_ch
                x = x.bmm(lamb_att).squeeze(1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        if self.DANN and not test:
            return out,barrier_out
        else:
            return out 


class Resnest(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Resnest, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = resnest101(pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.fc.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class Seresnext(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Seresnext, self).__init__()
        self.n_meta_features = n_meta_features
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.last_linear.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.last_linear = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
