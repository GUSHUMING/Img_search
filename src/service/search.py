import logging
from common.config import default_cache_dir
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from diskcache import Cache
import time
from torch_model.vgg16 import extract_feat


def query_name_from_ids(vids):
    res = []
    cache = Cache(default_cache_dir)
    for i in vids:
        if i in cache:
            res.append(cache[i])
    return res


def do_search(table_name, img_path, top_k, model):
    try:
        feats = []
        time1 = time.time()
        index_client = milvus_client()
        time2 = time.time()
        feat = extract_feat(model, img_path)
        # print(f'feat:{feat}')
        time3 = time.time()
        feats.append(feat)

        _, vectors = search_vectors(index_client, table_name, feats, top_k)
        time4 = time.time()
        vids = [x.id for x in vectors[0]]
        # print(f'vid:{vids}')
        # res = [x.decode('utf-8') for x in query_name_from_ids(vids)]

        # res_id = [x for x in query_name_from_ids(vids)]
        res_id = vids
        # print(res_id)
        res_distance = [x.distance for x in vectors[0]]
        # print(f'resid:{res_id}, resdistance:{res_distance}')
        time5 = time.time()
        # res = dict(zip(res_id,distance))
        print(f'\tsearch_start:{time1}\n\tconnect_client:{time2}\n\textract:{time3}\n\tsearch:{time4}\n\tcache:{time5}')

        return res_id, res_distance
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)
