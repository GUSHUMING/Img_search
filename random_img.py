import os
import random
import shutil
import re
dict_path = '/home/gushuming/testdata'
path = '/home/gushuming/imgdata'
path_new = os.path.join(dict_path, 'detection')
if not os.path.exists(path_new):
    os.makedirs(path_new)

# print(os.listdir(path))
for file in os.listdir(path):
    if re.match(r'130|270|1E',file):
        file_path = os.path.join(path, file, '0')
        if os.path.exists(file_path):
            pic_list = os.listdir(file_path)
            # print(pic_list)
            if len(pic_list) > 200:
                test_img = random.sample(pic_list, 10)
                for img in test_img:
                    print(img)
                    shutil.copyfile(os.path.join(file_path, img), os.path.join(path_new, img))
