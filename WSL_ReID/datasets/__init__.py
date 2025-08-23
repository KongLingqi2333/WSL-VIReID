import os

from .sysu import SYSU
from .llcm import LLCM
from .regdb import RegDB

_datasets = {
    'sysu': SYSU,
    'llcm': LLCM,
    'regdb': RegDB
}

def create(args):
    if args.dataset not in _datasets:
        raise KeyError("Unknown dataset:", args.dataset)
    print('building {} dataset ...'.format(args.dataset))
    return _datasets[args.dataset](args)
