import os
import argparse
import setproctitle
import torch
import warnings

import time
import datasets
import models
from task import train, test
from wsl import CMA
from utils import time_now, makedir, Logger, MultiItemAverageMeter, set_seed, save_checkpoint

warnings.filterwarnings("ignore")

def main(args):
    best_rank1 = 0
    best_mAP = 0
    log_path = os.path.join(args.save_path, "log/")
    model_path = os.path.join(args.save_path, "models/")
    makedir(log_path)
    makedir(model_path)
    logger = Logger(os.path.join(log_path, "log.txt"))
    if not args.resume and args.mode == 'train':
        logger.clear()
    logger(args)
    dataset = datasets.create(args)
    model = models.create(args)

    if args.mode == "train":
        cma = CMA(args)
        if args.resume or not args.model_path == 'default':
            enable_phase1 = False
            if 'wsl' in args.debug and not args.model_path == 'default':
                model.resume_model(args.model_path)
            else:
                model.resume_model()
        elif 'wsl' in args.debug:
            enable_phase1 = True
            model.resume_model()
        else:
            enable_phase1 = False
            model.resume_model()

        if enable_phase1:
            logger('Time: {} | start phase1 from epoch 0'.format(time_now()))
            for current_epoch in range(0, args.stage1_epoch):
                model.scheduler_phase1.step(current_epoch)
                _, result = train(args, model, dataset, current_epoch,cma,logger,enable_phase1)
                cmc, mAP, mINP = test(args, model, dataset,current_epoch) 
                best_rank1 = max(cmc[0], best_rank1)
                best_mAP = max(mAP, best_mAP)
                logger('Time: {} | phase 1 epoch {}; Setting: {}'.format(time_now(), current_epoch+1, args.save_path))
                logger(f'e_lr: {model.scheduler_phase1.get_lr()[0]}')
                logger(result)
                logger('R1:{:.4f};   R10:{:.4f};  R20:{:.4f};  mAP:{:.4f};  mINP:{:.4f}\n\
                       Best_R1: {:.4f};    Best mAP: {:.4f}'\
                       .format(cmc[0], cmc[9], cmc[19],mAP, mINP,best_rank1,best_mAP))
                logger('=================================================')
                if current_epoch == args.stage1_epoch-1:
                    save_checkpoint(args,model,current_epoch+1)
        enable_phase1 = False
        start_epoch = model.resume_epoch
        logger('Time: {} | start phase2 from epoch {}'.format(time_now(), start_epoch))
        for current_epoch in range(start_epoch, args.stage2_epoch):
            model.scheduler_phase2.step(current_epoch)
            _, result = train(args, model, dataset, current_epoch,cma,logger,enable_phase1)

            cmc, mAP, mINP = test(args, model, dataset,current_epoch) 
            is_best_rank = (cmc[0] >= best_rank1)
            is_best_mAP = (mAP >= best_mAP)
            best_rank1 = max(cmc[0], best_rank1)
            best_mAP = max(mAP, best_mAP)
            model.save_model(current_epoch, is_best_rank)
            logger('=================================================\nEpoch: {};Time: {};Setting: {}'\
                   .format(current_epoch, time_now(), args.save_path))
            logger(f'e_lr: {model.scheduler_phase2.get_lr()[0]}')
            logger(result)
            logger('R1:{:.4f};   R10:{:.4f};  R20:{:.4f};  mAP:{:.4f};  mINP:{:.4f}\n\
                   Best_R1: {:.4f};    Best mAP: {:.4f}'\
                   .format(cmc[0], cmc[9], cmc[19],mAP, mINP,best_rank1,best_mAP))
            logger('=================================================')
        
    if args.mode == 'test':
        if args.model_path == 'default':
            model.resume_model()
        else:
            model.resume_model(args.model_path)
        cmc, mAP, mINP = test(args, model, dataset)
        logger('Time: {}; Test on Dataset: {}'.format(time_now(), args.dataset))
        logger('R1:{:.4f};   R10:{:.4f};  R20:{:.4f};  mAP:{:.4f};  mINP:{:.4f}\n\
               Best_R1: {:.4f};    Best mAP: {:.4f}'\
               .format(cmc[0], cmc[9], cmc[19],mAP, mINP,best_rank1,best_mAP))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSL-ReID")
    parser.add_argument("--dataset", default="regdb", type=str, help="dataset name: sysu, llcm, regdb")
    parser.add_argument("--arch", default="resnet", type=str, help="sysu:clip-resnet, llcm:resnet, regdb:resnet")
    parser.add_argument('--mode', default='train', help='train or test')

    parser.add_argument("--data-path", default="../../VIREID_Dataset/", type=str, help="dataset path")
    parser.add_argument("--save-path", default="save/", type=str, help="log and model save path")

    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate 0.0003 for sysu(s) and llcm(l), 0.00045 for regdb(r)')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight deacy')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 70],
                        help='milestones for the learning rate decay, s&l:30, 70, r:50, 70')
    parser.add_argument('--relabel', default=1, type=int, help='relabel train dataset')

    parser.add_argument('--weak-weight',default=0.25, type=float, help='weight of weak loss')
    parser.add_argument('--tri-weight',default=0.25, type=float, help='weight of triplet loss')
    parser.add_argument('--img-h',default=288,type=int,help='height of the input image')
    parser.add_argument('--img-w',default=144,type=int,help='width of the input image')
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument('--num-workers', default=8, type=int, help='num workers')
    parser.add_argument('--batch-pidnum', default=8, type=int ,help='training pid in each batch, r:5')
    parser.add_argument('--pid-numsample',default=4, type=int ,help='num sample of each pid in a batch')
    parser.add_argument('--test-batch', default=128, type=int, metavar='tb', help='testing batch size')
    
    parser.add_argument('--sigma', default=0.8, type=float, help='momentum update factor')
    parser.add_argument('-T', '--temperature', default=3, type=float, help='Temperature parameter of softmax')
    parser.add_argument("--device", default=0, type=int, help="gpu")
    parser.add_argument('--stage1-epoch' ,default=20, type=int,help='s:20, l:80, r:50')
    parser.add_argument('--stage2-epoch' ,default=120, type=int,help='CMCL total epoch')
    parser.add_argument('--resume', default= 0, type = int, help='resume or not')
    parser.add_argument('--debug', default='wsl',type=str,help='wsl or sl')
    parser.add_argument('--trial', default=1,type=int,help='trial for regdb')
    parser.add_argument('--search-mode', default='all',type=str,help='all or indoor search gallery')
    parser.add_argument('--gall-mode', default='single',type=str,help='mutil or single shot')
    parser.add_argument('--test-mode', default='t2v',type=str,help='regdb and llcm test_mode')
    parser.add_argument('--model-path', default='default', type=str, help='load from checkpoint')
    args = parser.parse_args()
    args.save_path = '../saved_'+args.dataset+'_{}'.format(args.arch)+'/'+args.save_path
    if args.dataset =='sysu':
        args.num_classes = 395
    elif args.dataset =='regdb':
        args.num_classes = 206
        args.save_path += f'_{args.trial}'
    elif args.dataset == 'llcm':
        args.num_classes = 713
    set_seed(args.seed)
    setproctitle.setproctitle(args.save_path)
    main(args)