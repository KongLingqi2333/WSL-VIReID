import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import copy
import re

from .data_process import *
class LLCM:
    def __init__(self, args):
        self.args = args
        self.test_mode = args.test_mode

        self.path = os.path.join(args.data_path, "LLCM/")
        self.num_workers = args.num_workers
        self.pid_numsample = args.pid_numsample
        self.batch_pidnum = args.batch_pidnum
        self.batch_size = self.pid_numsample*self.batch_pidnum
        self.test_batch = args.test_batch

        self.train_rgb = LLCM_train(args, self.path, modal ='rgb')
        self.train_ir = LLCM_train(args, self.path, modal ='ir')

        self.rgb_relabel_dict = self.train_rgb.relabel_dict
        self.ir_relabel_dict = self.train_ir.relabel_dict
        
        self.gallery_list = []
        if self.test_mode == 'v2t':
            self.query = LLCM_test(args, self.path, 'rgb')    
            for i in range(10):
                gallery = LLCM_test(args, self.path, 'ir',trial = i)
                self.gallery_list.append(gallery)
        elif self.test_mode == 't2v':
            self.query = LLCM_test(args, self.path, 'ir')
            for i in range(10):
                gallery = LLCM_test(args, self.path, 'rgb',trial = i) 
                self.gallery_list.append(gallery)
        self.n_query = len(self.query)
        self.n_gallery = len(self.gallery_list[0]) 
        self._get_query_loader()
        self._get_gallery_loader()

    def get_train_loader(self):
        self.train_rgb.load_mode = 'train'
        self.train_ir.load_mode = 'train'
        sampler = LLCM_Sampler(self.args, self.train_rgb.label, self.train_ir.label)
        self.train_rgb.sampler_idx = sampler.rgb_index
        self.train_ir.sampler_idx = sampler.ir_index
        train_rgb_loader = data.DataLoader(self.train_rgb, batch_size=self.batch_size,\
                                           sampler=sampler, num_workers=self.num_workers, drop_last=True)
        train_ir_loader = data.DataLoader(self.train_ir, batch_size=self.batch_size,\
                                          sampler=sampler, num_workers=self.num_workers, drop_last=True)
        return train_rgb_loader, train_ir_loader
    
    def get_normal_loader(self):
        self.train_rgb.load_mode = 'test'
        self.train_ir.load_mode = 'test'
        normal_rgb_loader = data.DataLoader(self.train_rgb, batch_size=self.test_batch,
                                       num_workers=self.num_workers, drop_last=False)
        normal_ir_loader = data.DataLoader(self.train_ir, batch_size=self.test_batch,
                                       num_workers=self.num_workers, drop_last=False)
        return normal_rgb_loader, normal_ir_loader
    
    def _get_query_loader(self):
        query_loader = data.DataLoader(
            self.query, self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.query_loader  = query_loader
    def _get_gallery_loader(self):
        self.gall_info = []
        self.gallery_loaders = []
        for i in range(10):
            self.gall_info.append((self.gallery_list[i].test_label, self.gallery_list[i].test_cam))
            gallery_loader = data.DataLoader(
                self.gallery_list[i], self.test_batch, shuffle=False, num_workers=self.num_workers, drop_last=False)
            self.gallery_loaders.append(gallery_loader)

class LLCM_train(data.Dataset):
    def __init__(self, args, data_path, trial=None,  modal = None):
        self.num_classes = args.num_classes
        self.relabel = args.relabel
        self.data_path = data_path
        self.transform_color_normal = transform_color_normal
        self.transform_color_sa = transform_color_ca
        self.transform_infrared_normal = transform_infrared_normal
        self.transform_infrared_sa = transform_infrared_sa
        self.transform_test = transform_test
        self.modal = modal
        self.sampler_idx = None
        self.load_mode = None

        self.trial = trial
        self._init_data()

    def _init_data(self):
        if self.modal == 'rgb':
            train_list = self.data_path + 'idx/train_vis.txt'
            img_file, train_label,img_camid = self._load_data(train_list)
        elif self.modal == 'ir':
            train_list = self.data_path + 'idx/train_nir.txt'
            img_file, train_label,img_camid = self._load_data(train_list)
        else:
            raise ValueError("modal should be rgb or ir")
        train_image = []
        for i in range(len(img_file)):
            img = Image.open(self.data_path + img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_image.append(pix_array)

        train_image = np.array(train_image)
        length = len(train_label)
        train_label = np.array(train_label).reshape(length,1)
        train_idx = np.array([i for i in range(length)]).reshape(length,1)
        img_camid = np.array(img_camid).reshape(length,1)

        train_info = np.concatenate((train_idx,train_label,img_camid),axis=1)
        self.train_info, self.relabel_dict = self._relabel(train_info)
        self.train_image = train_image
        self.label = self.train_info[:,1]
        
    def _load_data(self,input_data_path):
        with open(input_data_path) as f:
            data_file_list = open(input_data_path, 'rt').read().splitlines()
            # Get full list of image and labels
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]
            pattern = re.compile(r'_c(\d+)')
            file_camid = [int(pattern.search(path).group(1)) for path in file_image]
        return file_image, file_label, file_camid
    
    def _relabel(self,info):
        # shuffle pids to erase corresponding relationship between modalities
        label = info[:,1]
        gt = label.reshape(label.shape[0],-1)
        info = np.concatenate((info,gt),axis=1)
        pid_set = set(label)
        random_pid = list(range(len(pid_set)))
        random.shuffle(random_pid)
        pid2label = {pid:idx for idx, pid in enumerate(pid_set)}
        pid2random_label = {pid:idx for idx, pid in enumerate(random_pid)}
        for i in range(len(label)):
            labeled_id = pid2label[label[i]]
            info[:,-1][i] = labeled_id
            if self.relabel:
                info[:,1][i]= pid2random_label[labeled_id]
            else:
                info[:,1][i]= labeled_id
        return info, pid2random_label
    
    def __len__(self):
        return len(self.train_image)
    
    def __getitem__(self, index):
        if self.load_mode == 'train': # get item for train
            idx = self.sampler_idx[index] # sampler index
            info = self.train_info[idx]
            img = self.train_image[idx]
            if self.modal == 'rgb':
                color_img = self.transform_color_normal(img)
                ca_img = self.transform_color_sa(img)
                return color_img, ca_img, info
            elif self.modal == 'ir':
                ir_img = self.transform_infrared_normal(img)
                aug_img = self.transform_infrared_sa(img)
                return ir_img, aug_img, info
            else:
                raise ValueError('invalid self.modal!')
            
        else: # get item for extrcat feature to match
            if self.modal == 'rgb':
                ori_img = self.transform_test(self.train_image[index])
                imgs = (ori_img)
            elif self.modal == 'ir':
                ori_img = self.transform_test(self.train_image[index])
                imgs = (ori_img)
            info = self.train_info[index]
            return imgs, info
        
class LLCM_test(data.Dataset):
    def __init__(self, args, data_path, modal, trial=-1):
        self.data_path = data_path
        self.modal = modal
        self.search_mode = args.search_mode
        self.gall_mode = args.gall_mode
        self.transform = transform_test
        test_img_file, test_label, test_cam = self._process_test_llcm(trial)
        
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            try:
                img = img.resize((args.img_w, args.img_h), Image.ANTIALIAS)
            except AttributeError:
                img = img.resize((args.img_w,args.img_h), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)

        self.test_image = test_image
        self.test_label = test_label
        self.test_cam = test_cam

    def __len__(self):
        return len(self.test_label)

    def __getitem__(self, index):
        return self.transform(self.test_image[index]), self.test_label[index]

    def _process_test_llcm(self,seed):
        if seed == -1:
            random.seed(0)
        else:
            random.seed(seed)
        if self.modal == "rgb":
            cameras = ['test_vis/cam1','test_vis/cam2',\
                       'test_vis/cam3','test_vis/cam4',\
                        'test_vis/cam5','test_vis/cam6',\
                        'test_vis/cam7','test_vis/cam8',\
                        'test_vis/cam9']
        elif self.modal == "ir":
            cameras = ['test_nir/cam1','test_nir/cam2',\
                       'test_nir/cam4','test_nir/cam5',\
                        'test_nir/cam6','test_nir/cam7',\
                        'test_nir/cam8','test_nir/cam9']
        file_path = os.path.join(self.data_path,'idx/test_id.txt')
        files = []
        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]
        for id in sorted(ids):
            for cam in cameras:
                img_dir = os.path.join(self.data_path,cam,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    if seed == -1:
                        files.extend(new_files)
                    else:
                        files.append(random.choice(new_files))
        test_img = []
        test_id = []
        test_cam = []
        for img_path in files:
            camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
            test_img.append(img_path)
            test_id.append(pid)
            test_cam.append(camid)
        return test_img, np.array(test_id), np.array(test_cam)
    
class LLCM_Sampler(data.Sampler):
    def __init__(self, args, rgb, ir):
        
        self.rgb = rgb
        self.ir = ir
        self.len=max(len(rgb),len(ir))
        self.num_classes = args.num_classes
        self.batch_pidnum = args.batch_pidnum
        self.pid_numsample = args.pid_numsample
        
        self.rgb_dict = {k:[] for k in range(self.num_classes)}
        self.ir_dict  = {k:[] for k in range(self.num_classes)}
        # position of rgb and ir images
        for i in range(len(rgb)):
            self.rgb_dict[int(rgb[i])].append(i)
            if i < len(ir):
                self.ir_dict[int(ir[i])].append(i)

        self._sampler()
        
    def _sampler(self):
        rgb_index = []
        ir_index = []
    
        batch_num = int(1+self.len/(self.batch_pidnum*self.pid_numsample))
        for i in range(batch_num):
            selected_id = random.sample(list(range(self.num_classes)),self.batch_pidnum)
            for each_id in selected_id:
                if min(len(self.rgb_dict[each_id]),len(self.ir_dict[each_id])) < self.pid_numsample:
                    selected_rgb = random.choices(self.rgb_dict[each_id],k=self.pid_numsample)
                    selected_ir = random.choices(self.ir_dict[each_id],k=self.pid_numsample)
                else:
                    selected_rgb = random.sample(self.rgb_dict[each_id],self.pid_numsample)
                    selected_ir = random.sample(self.ir_dict[each_id],self.pid_numsample)
                rgb_index.extend(selected_rgb)
                ir_index.extend(selected_ir)
        
        self.rgb_index = rgb_index
        self.ir_index = ir_index

    def __iter__(self):
        return iter(range(self.len))
            
    def __len__(self):
        return self.len