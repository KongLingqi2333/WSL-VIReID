import numpy as np
from PIL import Image
import os
import re
import tqdm
import pickle

IMAGE_HEIGHT = 288
IMAGE_WIDTH = 144
# data_path = "/media/d/KLQ/MINI_Dataset/SYSU-MM01/"
# data_path = '/home/ps/data/data-1/KLQ/MINI_Dataset/SYSU-MM01'

data_path = "/home/l/data_1/KLQ/VIREID_Dataset/SYSU-MM01/"
assert os.path.exists(data_path)
'''
save trainset and info as numpy array
'''

rgb_cameras = ["cam1", "cam2", "cam4", "cam5"]
ir_cameras = ["cam3", "cam6"]

# get train ids
file_path_train = os.path.join(data_path, "exp/train_id.txt")
file_path_val = os.path.join(data_path, "exp/val_id.txt")

with open(file_path_train, "r") as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(",")]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, "r") as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(",")]
    id_val = ["%04d" % x for x in ids]

id_train.extend(id_val)  # add val to train
files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + "/" + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + "/" + i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)


# save samples info
def read_sample(paths, modal=None):
    if modal == None:
        raise AttributeError("modal must be set")
    sample_info = []
    img_arrary = []
    index2path = {}
    # path_iter = tqdm.tqdm(enumerate(paths))
    path_iter = tqdm.tqdm(enumerate(paths), total=len(paths), desc=f"Processing {modal} paths")
    for index, path in path_iter:

        pattern = re.compile(r"cam([\d])/([\d]+)") # match camid，pid
        camid, pid = map(int, pattern.search(path).groups())  
        img = Image.open(path)
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        train_img = np.array(img)
        sample_info.append([index, pid, camid]) #<===================================================
        img_arrary.append(train_img)
        index2path[index] = path
    with open(os.path.join(data_path, f"{modal}i2p.pk"),'wb') as f:
        pickle.dump(index2path,f)
    # np.save(os.path.join(data_path, f"{modal}i2p.npy"), index2path)
    return np.array(sample_info), np.array(img_arrary)


rgb_train_info, rgb_train = read_sample(files_rgb,modal='rgb')
ir_train_info, ir_train = read_sample(files_ir,modal='ir')

np.save(os.path.join(data_path,"train_rgb_modified_img.npy"), rgb_train)
np.save(os.path.join(data_path,"train_rgb_info.npy"), rgb_train_info)
np.save(os.path.join(data_path,"train_ir_modified_img.npy"), ir_train)
np.save(os.path.join(data_path,"train_ir_info.npy"), ir_train_info)
