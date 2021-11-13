import os
import time
from diskcache import Cache
import cv2
import torch
import torchvision.transforms as transforms
from numpy import linalg as LA
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import re
from collect.collect import collate_fn
from common.config import default_cache_dir
from indexer.index import milvus_client, create_table, insert_vectors, create_index, \
    has_table,count_table
from torch_model.vgg16 import vgg16_net

torch.cuda.set_device(1)
id2path_io = open('./id2path.txt', mode='w')
cache = Cache(default_cache_dir, size_limit=1024 * 1024 * 1024 * 15)



def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.png'))]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def to_var(x):
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class imgdata(Dataset):
    def __init__(self, img_path_list, transform):
        cv2.setNumThreads(0)
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):

        img_path = self.img_path_list[idx]
        img = cv2.imread(img_path, 1)
        try:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            # print(type(img))
        except:
            return None, img_path
            # img = np.ones((244, 244, 3))
            # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = self.transform(img)
        img = img.float()
        return img, img_path


def load_imgs(table_name, file_path, model):
    cats = []
    for cat in os.listdir(file_path):

        if cat:
            cats.append(cat)
            cat_path = os.path.join(file_path, cat)
            print(cat_path)
            do_load(table_name, cat_path, model)

    ## create index finally
    index_client = milvus_client()
    create_index(index_client, table_name)


def do_load(table_name, class_path, model):
    print(f'Start cat:{class_path} at:{now()}')

    index_client = milvus_client()

    status, ok = has_table(index_client, table_name)
    if not ok:
        print("create table.")
        create_table(index_client, table_name=table_name)
        create_index(index_client, table_name)

    img_folder_indexs = os.listdir(class_path)
    img_folder_indexs.sort(key=int)
    img_folders = [os.path.join(class_path, index) for index in img_folder_indexs]

    img_list = []

    for img_folder in img_folders:
        print(f'\tStart folder: {img_folder} at:{now()}', flush=True)

        folder_img_list = get_imlist(img_folder)

        for img in folder_img_list:
            img_list.append(img)

            if len(img_list) >= 5120:
                dealBulkImg(img_list, model, cache, index_client, table_name)
                img_list = []
    if img_list:
        dealBulkImg(img_list, model, cache, index_client, table_name)
    print(f'Finish cat:{class_path}   at:{now()}\n')


def dealBulkImg(img_list, model, cache, index_client, table_name):
    Feats = []
    Name = []
    transf = transforms.ToTensor()
    dataset = imgdata(img_list, transf)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=16)
    for img, name in dataloader:
        print(f'start batch {now()}')
        img = to_var(img)
        feat = model.forward(img)
        feat = feat.data.cpu().numpy()
        feat = feat.squeeze()
        name = list(name)
        Name = Name + name
        if feat.ndim == 1:
            norm_feat = feat / LA.norm(feat)
            norm_feat = [i.item() for i in norm_feat]
            Feats.append(norm_feat)
        else:
            for f in feat:
                norm_feat = f / LA.norm(f)
                norm_feat = [i.item() for i in norm_feat]
                Feats.append(norm_feat)
        # time.sleep(0.1)
    status, ids = insert_vectors(index_client, table_name, Feats)
    for i in range(len(Name)):
        img_path = Name[i]
        cache[ids[i]] = img_path
        id2path_io.write('{}\t{}\n'.format(ids[i], img_path))
    time.sleep(1)
    # print('cache',len(cache))
    # print('n',count_table(index_client,table_name))



if __name__ == "__main__":
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    model = vgg16_net()
    model = model.eval()
    model = model.cuda()
    table_name = "imgs_all"
    database_path = '/mnt/weeddata/imgs'
    load_imgs(table_name, database_path, model)

