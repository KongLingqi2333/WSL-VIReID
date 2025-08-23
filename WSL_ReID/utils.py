import os
import sys
import random
# from typing import Any
import numpy as np
import torch
import time

def save_checkpoint(args, model, epoch):
    if args.dataset == 'regdb':
        path = '../saved_pretrain_{}_{}_{}_{}/'.format(args.dataset,args.arch,args.trial,args.save_path)
    else:
        path = '../saved_pretrain_{}_{}/'.format(args.dataset,args.arch)
    makedir(path)
    all_state_dict = {'backbone': model.model.state_dict(),
                    'classifier1': model.classifier1.state_dict(), 
                    'classifier2': model.classifier2.state_dict(),
                    'classifier3': model.classifier3.state_dict()}
    torch.save(all_state_dict,path+'model_{}.pth'.format(epoch))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('make dir {} successful!'.format(path))

def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # defualt deterministic = True, benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

@torch.no_grad()
def infoEntropy(input):
    input = torch.nn.functional.softmax(input,dim=1)
    output = -torch.mean(input*torch.log2(input))
    return output

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def __call__(self, input):
        input = str(input)
        with open(self.log_file, 'a') as f:
            f.writelines(input+'\n')
        print(input)

    def clear(self):
        with open(self.log_file, 'w') as f:
            pass


class MultiItemAverageMeter:
    def __init__(self):
        self.content = {}

    def update(self, val):
        for key in list(val.keys()):
            value = val[key]
            if key not in list(self.content.keys()):
                self.content[key] = {'avg': value, 'sum': value, 'count': 1.0}
            else:
                self.content[key]['sum'] += value
                self.content[key]['count'] += 1.0
                self.content[key]['avg'] = self.content[key]['sum'] / self.content[key]['count']

    def get_val(self):
        keys = list(self.content.keys())
        values = []
        for key in keys:
            try:
                values.append(self.content[key]['avg'].data.cpu().numpy())
            except:
                values.append(self.content[key]['avg'])
        return keys, values

    def get_str(self):
        result = ''
        keys, values = self.get_val()

        for i,(key, value) in enumerate(zip(keys, values)):
            result += key
            result += ': '
            result += str(value)
            result += ';  '
            if i%2:
                result += '\n'

        return result

def pha_unwrapping(x):
    fft_x = torch.fft.fft2(x.clone(), dim=(-2, -1))
    fft_x = torch.stack((fft_x.real, fft_x.imag), dim=-1)
    pha_x = torch.atan2(fft_x[:, :, :, :, 1], fft_x[:, :, :, :, 0])

    fft_clone = torch.zeros(fft_x.size(), dtype=torch.float).cuda()
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x.clone())
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x.clone())

    # get the recomposed image: source content, target style
    pha_unwrap = torch.fft.ifft2(torch.complex(fft_clone[:, :, :, :, 0], fft_clone[:, :, :, :, 1]),
                                 dim=(-2, -1)).float()

    return pha_unwrap.to(x.device)