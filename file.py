# coding:utf-8

import os
import sys
import getpass
import shutil

database_path = '/mnt/weeddata/imgs'
path = '/home/gushuming/imgdata'
if not os.path.exists(path):
    os.mkdir(path)
for cat in os.listdir(database_path):
    if cat:
        try:
            cat_path = os.path.join(database_path, cat, '0')
            # print(cat)
            target_path = os.path.join(path, cat)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            target_path_ = os.path.join(target_path, '0')
            shutil.copytree(cat_path, target_path_)
        except Exception as e:
            print(e)
