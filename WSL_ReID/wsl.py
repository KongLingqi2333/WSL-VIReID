import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter, OrderedDict
from sklearn.preprocessing import normalize
import time
import pickle
from utils import fliplr
class CMA(nn.Module):
    '''
    Cross modal Match Aggregation
    '''
    def __init__(self, args):
        super(CMA, self).__init__()
        # self.inited = False
        self.device = torch.device(args.device)
        self.not_saved = True
        # self.threshold = 0.8
        self.num_classes = args.num_classes
        self.T = args.temperature # softmax temperature
        self.sigma = args.sigma # momentum update factor
        # memory of visible and infrared modal
        self.register_buffer('vis_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('ir_memory',torch.zeros(self.num_classes,2048))

    @torch.no_grad()
    # def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode):
    def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode, rgb_features=None, ir_features=None):
    # vis: vis sample(v2i scores or vis features) ir: ir sample
        self.mode = mode
        self.not_saved = False
        if self.mode != 'scores' and self.mode != 'features':
            raise ValueError('invalid mode!')
        elif self.mode == 'scores': # predict scores
            vis = torch.nn.functional.softmax(self.T*vis,dim=1)
            ir = torch.nn.functional.softmax(self.T*ir,dim=1)
        ###############################
        # save features in memory bank
        if rgb_features is not None and ir_features is not None:
            # Prepare empty memory banks on the device
            self.vis_memory = self.vis_memory.to(self.device)
            self.ir_memory = self.ir_memory.to(self.device)
            
            # Get unique labels and process RGB and IR features
            label_set = torch.unique(rgb_ids)
            
            for label in label_set:
                # Select RGB features for the current label
                rgb_mask = (rgb_ids == label)
                ir_mask = (ir_ids == label)
                # .any() check True in bool tensor
                if rgb_mask.any():
                    rgb_selected = rgb_features[rgb_mask]
                    self.vis_memory[label] = rgb_selected.mean(dim=0)
                
                if ir_mask.any():
                    ir_selected = ir_features[ir_mask]
                    self.ir_memory[label] = ir_selected.mean(dim=0)
        ################################
        vis = vis.detach().cpu().numpy()
        ir = ir.detach().cpu().numpy()
        rgb_ids, ir_ids = rgb_ids.cpu(), ir_ids.cpu()
            
        self.vis, self.ir = vis, ir
        self.rgb_ids, self.ir_ids = rgb_ids, ir_ids
        self.rgb_idx, self.ir_idx = rgb_idx, ir_idx
        
    @torch.no_grad()
    def update(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        rgb_set = torch.unique(rgb_labels)
        ir_set = torch.unique(ir_labels)
        for i in rgb_set:
            rgb_mask = (rgb_labels == i)
            selected_rgb = rgb_feats[rgb_mask].mean(dim=0)
            self.vis_memory[i] = (1-self.sigma)*self.vis_memory[i] + self.sigma * selected_rgb
        for i in ir_set:
            ir_mask = (ir_labels == i)
            selected_ir = ir_feats[ir_mask].mean(dim=0)
            self.ir_memory[i] = (1-self.sigma)*self.ir_memory[i] + self.sigma * selected_ir

    def get_label(self, epoch=None):
        if self.not_saved:# pass if 
            pass
        else:
            print('get match labels')
            if self.mode == 'features':
                dists = np.matmul(self.vis, self.ir.T)
                v2i_dict, i2v_dict = self._get_label(dists,'dist')

            elif self.mode == 'scores':
                v2i_dict, _ = self._get_label(self.vis,'rgb')
                i2v_dict, _ = self._get_label(self.ir,'ir')
                self.v2i = v2i_dict
                self.i2v = i2v_dict
            return v2i_dict, i2v_dict
    # TODO
    def _get_label(self,dists,mode):
        sample_rate = 1
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]# flat to 1d and sort
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)# sort index return to 2d, like ([0,1,2],[1,2,0])
        idx1, idx2 = sorted_2d[0], sorted_2d[1]# sorted idx of dim0 and dim1
        dists = dists[idx1, idx2]
        idx_length = int(np.ceil(sample_rate*dists.shape[0]/self.num_classes))
        dists = dists[:idx_length]

        if mode=='dist': # multiply the instance features of the two modalities
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                            np.array(self.ir_ids)[idx2[:idx_length]])]
            
        elif mode=='rgb': # classify score of RGB (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]

        elif mode=='ir': # classify score of IR (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.ir_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]
        else:
            raise AttributeError('invalid mode!')
        convert_label_cnt = Counter(convert_label)
        convert_label_cnt_sorted = sorted(convert_label_cnt.items(),key = lambda x:x[1],reverse = True)
        length = len(convert_label_cnt_sorted)
        lambda_cm=0.1
        in_rgb_label=[]
        in_ir_label=[]
        v2i = OrderedDict()
        i2v = OrderedDict()

        length_ratio = 1
        for i in range(int(length*length_ratio)):
            key = convert_label_cnt_sorted[i][0] 
            value = convert_label_cnt_sorted[i][1]
            # if key[0] == -1 or key[1] == -1:
            #     continue
            if key[0] in in_rgb_label or key[1] in in_ir_label:
                continue
            in_rgb_label.append(key[0])
            in_ir_label.append(key[1])
            v2i[key[0]] = key[1]
            i2v[key[1]] = key[0]
            # v2i[key[0]][key[1]] = 1
            
        return v2i, i2v # only v2i/i2v is used in scores mode

    def extract(self, args, model, dataset):
        '''
        Output: BN_features, labels, cls
        '''
        # save epoch
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader() 
        with torch.no_grad():
            
            rgb_features, rgb_labels, rgb_gt, r2i_cls, rgb_idx = self._extract_feature(model, rgb_loader,'rgb')
            ir_features, ir_labels, ir_gt, i2r_cls, ir_idx = self._extract_feature(model, ir_loader,'ir')

        # # //match by cls and save features to memory bank
        self.save(r2i_cls, i2r_cls, rgb_labels, ir_labels, rgb_idx,\
                 ir_idx, 'scores', rgb_features, ir_features)
        
    def _extract_feature(self, model, loader, modal):

        print('extracting {} features'.format(modal))

        saved_features, saved_labels, saved_cls= None, None, None
        saved_gts, saved_idx= None, None
        for imgs_list, infos in loader:
            labels = infos[:,1]
            idx = infos[:,0]
            gts = infos[:,-1].to(model.device)
            if imgs_list.__class__.__name__ != 'list':
                imgs = imgs_list
                imgs, labels, idx = \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            else:
                ori_imgs, ca_imgs = imgs_list[0], imgs_list[1]
                if len(ori_imgs.shape) < 4:
                    ori_imgs = ori_imgs.unsqueeze(0)
                    ca_imgs = ca_imgs.unsqueeze(0)

                imgs = torch.cat((ori_imgs,ca_imgs),dim=0)
                labels = torch.cat((labels,labels),dim=0)
                idx = torch.cat((idx,idx),dim=0)
                gts= torch.cat((gts,gts),dim=0).to(model.device)
                imgs, labels, idx= \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            _, bn_features = model.model(imgs) # _:gap feature

            if modal == 'rgb':
                cls, l2_features = model.classifier2(bn_features)
            elif modal == 'ir':
                cls, l2_features = model.classifier1(bn_features)
            l2_features = l2_features.detach().cpu()

            if saved_features is None: 
                # saved_features, saved_labels, saved_cls, saved_idx = l2_features, labels, cls, idx
                saved_features, saved_labels, saved_cls, saved_idx = bn_features, labels, cls, idx

                saved_gts = gts
            else:
                # saved_features = torch.cat((saved_features, l2_features), dim=0)
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)

                saved_gts = torch.cat((saved_gts, gts), dim=0)
        return saved_features, saved_labels, saved_gts, saved_cls, saved_idx