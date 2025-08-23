import torch.nn as nn
import torch

def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, val=0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias:
            nn.init.constant_(m.bias, val=0.0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
        if m.affine:
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0.0)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=1e-3)
        if m.bias:
            nn.init.constant_(m.bias, val=0.0)

class Normalize(nn.Module):
    def __init__(self,power = 2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, input):
        norm = input.pow(self.power).sum(dim=1, keepdim=True).pow(1 / self.power)
        output = input / norm
        return output

class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.p)
            + ", "
            + "output_size="
            + str(self.output_size)
            + ")"
        )

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """Same, but norm is trainable"""

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)

class Text_Classifier(nn.Module):
    def __init__(self,args):
        super(Text_Classifier, self, ).__init__()
        self.num_classes = args.num_classes
        self.BN = nn.BatchNorm1d(1024)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(1024, self.num_classes, bias=False)
        self.classifier.apply(weights_init)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        if self.training:
            return cls_score
        else:
            self.l2_norm(bn_features)
class Image_Classifier(nn.Module):
    '''
    **Output: x_score, x_l2**
    '''
    def __init__(self, args):
        super(Image_Classifier, self).__init__()
        self.num_classes = args.num_classes
        self.classifier = nn.Linear(2048, args.num_classes, bias = False)
        self.classifier.apply(weights_init)

        self.l2_norm = Normalize(2)

    def forward(self,x_bn):
        x_score = self.classifier(x_bn)
        return x_score, self.l2_norm(x_bn)


