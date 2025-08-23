import torch
from models import Model
from datasets import SYSU
import time
import numpy as np
import random
import copy
from collections import OrderedDict
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy,pha_unwrapping
from models import Model

def train(args, model: Model, dataset, *arg):
    epoch = arg[0]
    cma:CMA = arg[1]
    logger = arg[2]
    enable_phase1 = arg[3]

    # //get feature for match
    if 'wsl' in args.debug or not enable_phase1:
        cma.extract(args, model, dataset)        
        rgb_labeling_dict, ir_labeling_dict = \
            dataset.train_rgb.relabel_dict, dataset.train_ir.relabel_dict
        r2i_pair_dict, i2r_pair_dict = cma.get_label(epoch)
        common_dict, specific_dict, remain_dict = {},{},{}
        i2r_specific_dict, r2i_specific_dict, r2i_remain_dict, i2r_remain_dict = {},{},{},{}
        for r,i in r2i_pair_dict.items():
            if i in i2r_pair_dict.keys() and i2r_pair_dict[i] == r:
                common_dict[r] = i
            elif r not in i2r_pair_dict.values() and i not in i2r_pair_dict.keys():
                r2i_specific_dict[r] = i
                specific_dict[r] = i
            else:
                r2i_remain_dict[r] = i
                remain_dict[r] = i
        for i,r in i2r_pair_dict.items():
            if (r,i) in common_dict.items():
                continue
            elif r not in r2i_pair_dict.values() and i not in r2i_pair_dict.keys():
                i2r_specific_dict[i] = r
                specific_dict[r] = i
            else:
                i2r_remain_dict[i] = r
                remain_dict[r] = i

        all_rm = torch.zeros((args.num_classes,args.num_classes)).to(model.device) # all corresponding pairs
        common_rm = all_rm.clone() # common corresponding pairs
        specific_rm = all_rm.clone() # specific corresponding pairs
        remain_rm = all_rm.clone() # remain corresponding pairs
        r2i_rm = all_rm.clone() # r2i corresponding pairs
        i2r_rm = all_rm.clone() # i2r corresponding pairs
        for r, i in common_dict.items(): 
            common_rm[r,i] += 1
        for r, i in specific_dict.items():
            specific_rm[r,i] += 1
        for r, i in r2i_pair_dict.items():
            r2i_rm[r,i] += 1
        for i, r in i2r_pair_dict.items():
            i2r_rm[i,r] += 1
        for r, i in remain_dict.items():
            remain_rm[r,i] += 1

        specific_rm = specific_rm + common_rm
        matched_rgb, matched_ir = list(r2i_pair_dict.keys()), list(i2r_pair_dict.keys())
        common_matched_rgb, common_matched_ir = list(common_dict.keys()), list(common_dict.values())
        specific_matched_rgb, specific_matched_ir = list(specific_dict.keys()), list(specific_dict.values())
        remain_matched_rgb, remain_matched_ir = list(remain_dict.keys()), list(remain_dict.values())
        all_matched_rgb = list(set(common_matched_rgb + specific_matched_rgb + remain_matched_rgb))
        all_matched_ir = list(set(common_matched_ir + specific_matched_ir + remain_matched_ir))
        matched_rgb = torch.tensor(matched_rgb).to(model.device)
        matched_ir = torch.tensor(matched_ir).to(model.device)
        common_matched_rgb = torch.tensor(common_matched_rgb).to(model.device)
        common_matched_ir = torch.tensor(common_matched_ir).to(model.device)
        specific_matched_rgb = torch.tensor(specific_matched_rgb).to(model.device)
        specific_matched_ir = torch.tensor(specific_matched_ir).to(model.device)

        remain_matched_rgb = torch.tensor(remain_matched_rgb).to(model.device)
        remain_matched_ir = torch.tensor(remain_matched_ir).to(model.device)
        all_matched_rgb = torch.tensor(all_matched_rgb).to(model.device)
        all_matched_ir = torch.tensor(all_matched_ir).to(model.device)

        if not model.enable_cls3:
            model.enable_cls3 = True
    # ======================================================
    model.set_train()
    meter = MultiItemAverageMeter()
    bt = args.batch_pidnum*args.pid_numsample
    rgb_loader, ir_loader = dataset.get_train_loader()

    nan_batch_counter = 0
    for (rgb_imgs, ca_imgs, color_info), (ir_imgs, aug_imgs,ir_info) in zip(rgb_loader, ir_loader):
        if enable_phase1:
            model.optimizer_phase1.zero_grad()
        else:
            model.optimizer_phase2.zero_grad()
        rgb_imgs, ca_imgs = rgb_imgs.to(model.device), ca_imgs.to(model.device)

        color_imgs = torch.cat((rgb_imgs, ca_imgs), dim = 0)
        rgb_gts, ir_gts = color_info[:,-1], ir_info[:,-1] 
        rgb_ids, ir_ids = color_info[:,1], ir_info[:,1]
        rgb_ids = torch.cat((rgb_ids,rgb_ids)).to(model.device)
        if args.dataset == 'regdb':
            ir_imgs, aug_imgs = ir_imgs.to(model.device), aug_imgs.to(model.device)
            ir_imgs = torch.cat((ir_imgs, aug_imgs), dim = 0)
            ir_ids = torch.cat((ir_ids,ir_ids)).to(model.device)
        else:
            ir_imgs = ir_imgs.to(model.device)
            ir_ids = ir_ids.to(model.device)
        gap_features, bn_features = model.model(color_imgs, ir_imgs)
        rgbcls_out, _l2_features = model.classifier1(bn_features)
        ircls_out, _l2_features = model.classifier2(bn_features)

        rgb_features, ir_features = gap_features[:2*bt], gap_features[2*bt:]
        r2r_cls, i2i_cls, r2i_cls,i2r_cls =\
              rgbcls_out[:2*bt], ircls_out[2*bt:], ircls_out[:2*bt], rgbcls_out[2*bt:]
        if 'wsl' in args.debug:
            if enable_phase1:
                r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
                i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
                total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data,
                            'r2r_tri_loss':r2r_tri_loss.data,
                            'i2i_tri_loss':i2i_tri_loss.data})
            else:
                r2c_cls = model.classifier3(bn_features)[0][:2*bt]
                i2c_cls = model.classifier3(bn_features)[0][2*bt:]
                dtd_features = bn_features.detach()
                dtd_rgbcls_out = model.classifier1(dtd_features)[0]
                dtd_ircls_out = model.classifier2(dtd_features)[0]
                dtd_r2r_cls, dtd_i2r_cls = dtd_rgbcls_out[:2*bt], dtd_rgbcls_out[2*bt:]
                dtd_r2i_cls, dtd_i2i_cls = dtd_ircls_out[:2*bt], dtd_ircls_out[2*bt:]
                r2r_id_loss = model.pid_criterion(dtd_r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(dtd_i2i_cls, ir_ids)
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data})
                total_loss = r2r_id_loss + i2i_id_loss
                common_rgb_indices = torch.isin(rgb_ids, common_matched_rgb)
                common_ir_indices = torch.isin(ir_ids, common_matched_ir)
                ###############################################################
                if args.debug == 'wsl':
                    tri_rgb_indices = torch.isin(rgb_ids, common_matched_rgb)
                    tri_ir_indices = torch.isin(ir_ids, common_matched_ir)
                    selected_tri_rgb_ids = rgb_ids[tri_rgb_indices]
                    selected_tri_ir_ids = ir_ids[tri_ir_indices]
                    translated_tri_rgb_label = torch.nonzero(common_rm[selected_tri_rgb_ids])[:,-1]
                    translated_tri_ir_label = torch.nonzero(common_rm.T[selected_tri_ir_ids])[:,-1]
                
                    selected_tri_rgb_features = rgb_features[tri_rgb_indices]
                    selected_tri_ir_features = ir_features[tri_ir_indices]
                    matched_tri_rgb_features = torch.cat((selected_tri_rgb_features,ir_features),dim=0)
                    matched_tri_ir_features = torch.cat((rgb_features,selected_tri_ir_features),dim=0)
                    matched_tri_rgb_labels = torch.cat((translated_tri_rgb_label,ir_ids),dim=0)
                    matched_tri_ir_labels = torch.cat((rgb_ids,translated_tri_ir_label),dim=0)
                    tri_loss_rgb = args.tri_weight * model.tri_criterion(matched_tri_rgb_features, matched_tri_rgb_labels)
                    tri_loss_ir = args.tri_weight * model.tri_criterion(matched_tri_ir_features, matched_tri_ir_labels)
                    meter.update({'tri_loss_rgb':tri_loss_rgb.data,
                                'tri_loss_ir':tri_loss_ir.data})
                    total_loss += tri_loss_rgb + tri_loss_ir

                    selected_common_rgb_ids = rgb_ids[common_rgb_indices]
                    selected_common_ir_ids = ir_ids[common_ir_indices]
                    translated_cmo_rgb_label = torch.nonzero(common_rm[selected_common_rgb_ids])[:,-1]
                    translated_cmo_ir_label = torch.nonzero(common_rm.T[selected_common_ir_ids])[:,-1]
                    cma.update(bn_features[:2*bt], bn_features[2*bt:], rgb_ids, ir_ids)
                    r2i_entropy = infoEntropy(r2i_cls)
                    i2r_entropy = infoEntropy(i2r_cls)
                    w_r2i = r2i_entropy/(r2i_entropy+i2r_entropy)
                    w_i2r = i2r_entropy/(r2i_entropy+i2r_entropy)
                    selected_rgb_memory = cma.vis_memory[translated_cmo_ir_label].detach()
                    selected_ir_memory = cma.ir_memory[translated_cmo_rgb_label].detach()
                    mem_r2i_cls,_ = model.classifier2(selected_rgb_memory)
                    mem_i2r_cls,_ = model.classifier1(selected_ir_memory)
                    cmo_criterion = torch.nn.MSELoss()

                    if (selected_tri_ir_ids.shape[0]!=0):
                        r2i_cmo_loss = w_r2i * cmo_criterion(dtd_i2i_cls[common_ir_indices],mem_r2i_cls)
                        if torch.isnan(r2i_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'r2i_cmo_loss':r2i_cmo_loss.data})
                            total_loss += r2i_cmo_loss
                    if (selected_tri_rgb_ids.shape[0]!=0):
                        i2r_cmo_loss = w_i2r * cmo_criterion(dtd_r2r_cls[common_rgb_indices],mem_i2r_cls)
                        if torch.isnan(i2r_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'i2r_cmo_loss':i2r_cmo_loss.data})
                            total_loss += i2r_cmo_loss

                if epoch >= 30:
                    remain_rgb_indices = torch.isin(rgb_ids, remain_matched_rgb)
                    remain_ir_indices = torch.isin(ir_ids, remain_matched_ir)
                    remain_rgb_ids = rgb_ids[remain_rgb_indices]
                    remain_ir_ids = ir_ids[remain_ir_indices]
                    remain_r2c_cls = r2c_cls[remain_rgb_indices]
                    remain_i2c_cls = i2c_cls[remain_ir_indices]
                    if (remain_rgb_indices.shape[0]>0):
                        weak_r2c_loss = args.weak_weight*model.weak_criterion(remain_r2c_cls, remain_rm[remain_rgb_ids])
                        if torch.isnan(weak_r2c_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'weak_r2c_loss':weak_r2c_loss.data})
                            total_loss += weak_r2c_loss
        if enable_phase1:
            total_loss.backward()
            model.optimizer_phase1.step()
        else:                
            if args.debug == 'wsl':# //use modal specific pseudo labels
                specific_rgb_indices = torch.isin(rgb_ids, specific_matched_rgb)
                specific_ir_indices = torch.isin(ir_ids, specific_matched_ir)
                rgb_indices = specific_rgb_indices ^ common_rgb_indices
                ir_indices = specific_ir_indices ^ common_ir_indices

                selected_ir_ids = ir_ids[ir_indices]
                selected_rgb_ids = rgb_ids[rgb_indices]
                selected_i2c_cls = i2c_cls[ir_indices]
                selected_r2c_cls = r2c_cls[rgb_indices]

                if (selected_rgb_ids.shape[0]>0):
                    rgb_cross_loss = model.pid_criterion(selected_r2c_cls, specific_rm[selected_rgb_ids])
                    if torch.isnan(rgb_cross_loss).any():
                        nan_batch_counter+=1
                    else:
                        meter.update({'rgb_cross_loss':rgb_cross_loss.data})
                        total_loss += rgb_cross_loss
                ir_cross_loss = model.pid_criterion(i2c_cls, ir_ids)
                meter.update({'ir_cross_loss':ir_cross_loss.data})
                total_loss+= ir_cross_loss
                    
            elif args.debug == 'baseline':
                    r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                    i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                    r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
                    i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
                    total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
                    meter.update({'r2r_id_loss':r2r_id_loss.data,
                                'i2i_id_loss':i2i_id_loss.data,
                                'r2r_tri_loss':r2r_tri_loss.data,
                                'i2i_tri_loss':i2i_tri_loss.data})
            
            elif args.debug == 'sl':
                # //supervised learning
                rgb_gts = torch.cat((rgb_gts,rgb_gts)).to(model.device)
                ir_gts = torch.cat((ir_gts,ir_gts)).to(model.device)
                gts = torch.cat((rgb_gts,ir_gts))

                id_loss = model.pid_criterion(rgbcls_out, gts)
                tri_loss = model.tri_criterion(gap_features, gts)
                total_loss = id_loss + args.tri_weight*tri_loss
                meter.update({'id_loss': id_loss.data,
                                'tri_loss': tri_loss.data})

            else:
                raise RuntimeError('Debug mode {} not found!'.format(args.debug))
        
            total_loss.backward()
            model.optimizer_phase2.step()
    return meter.get_val(), meter.get_str()

def relabel(select_ids, source_labels, target_labels):
    '''
    Input: source_labels, target_labels
    Output: corresponding select_ids in target modal
    '''
    key_to_value = torch.full((torch.max(source_labels) + 1,), -1, dtype=torch.long).to(source_labels.device)
    key_to_value[source_labels] = target_labels
    
    select_ids = key_to_value[select_ids]
    return select_ids

def hate_nan(loss, condition,logger):
    if torch.isnan(loss):
        if condition:
            logger('no matched labels')
        else:
            logger('nan loss detected')
        return torch.tensor(0.0).to(loss.device)
    else:
        return loss