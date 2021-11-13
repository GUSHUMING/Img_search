import logging
import time
from indexer.index import milvus_client, search_vectors
from service.search import query_name_from_ids
import os
from torch_model.vgg16 import vgg16_net, extract_feat
import shutil


def do_search(table_name, top_k, feat):
    try:
        feats = []
        time1 = time.time()
        index_client = milvus_client()
        time2 = time.time()
        feats.append(feat)
        _, vectors = search_vectors(index_client, table_name, feats, top_k)
        time3 = time.time()
        vids = [x.id for x in vectors[0]]
        print(f'vid:{vids}')

        res_id = [x for x in query_name_from_ids(vids)]

        res_distance = [x.distance for x in vectors[0]]
        print(f'resid:{res_id}, resdistance:{res_distance}')
        time4 = time.time()

        print(
            f'\tconnect_client:{(time2 - time1) * 1000}ms\n\tsearch:{(time3 - time2) * 1000}ms\n\tcache:{(time4 - time3) * 1000}\n')

        return res_id, res_distance
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)


model = vgg16_net()
model = model.eval()
table_name = 'imgs_all'
path = '/home/gushuming/testdata/2'
result_path = os.path.join('/home/gushuming/testdata', "result")
if not os.path.exists(result_path):
    os.mkdir(result_path)
for i in os.listdir(path):
    name = os.path.splitext(i)[0]
    name_path = os.path.join(result_path, name)
    if not os.path.exists(name_path):
        os.mkdir(name_path)

    img_path = os.path.join(path, i)
    print(img_path)
    shutil.copyfile(img_path, os.path.join(name_path, 'test.' + os.path.splitext(i)[-1]))
    norm_feat = extract_feat(model, img_path)
    print('finish extract_feat')
    top_k = 10
    t1 = time.time()
    res_id, res_distance = do_search(table_name, top_k, norm_feat)
    t2 = time.time()

    print('time:%s ms' % ((t2 - t1) * 1000))
    for pic in res_id:
        shutil.copyfile(pic, os.path.join(name_path, os.path.split(pic)[-1]))
