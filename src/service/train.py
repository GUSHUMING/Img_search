import os
import logging
import time
from common.config import DEFAULT_TABLE
from common.config import default_cache_dir
# from common.config import DATA_PATH as database_path
from encoder.encode import feature_extract
from preprocessor.vggnet import VGGNet
from diskcache import Cache
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index,has_table


def do_train(table_name, database_path):
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(default_cache_dir)
    
    index_client = milvus_client()
    # delete_table(index_client, table_name=table_name)
    # time.sleep(1)
    status, ok = has_table(index_client, table_name)
    if not ok:
        print("create table.")
        create_table(index_client, table_name=table_name)

    img_folder_indexs = os.listdir(database_path)
    img_folder_indexs.sort(key = int)
    img_folders = [os.path.join(database_path, index) for index in img_folder_indexs]

    for img_folder in img_folders:
        try:
            vectors, names = feature_extract(img_folder, VGGNet())

            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cnt = len(vectors)
            print(f"insert into:{table_name} and img_foler:{img_folder} and cnt:{cnt} at:{now}")

            status, ids = insert_vectors(index_client, table_name, vectors)
            
            for i in range(len(names)):
                # cache[names[i]] = ids[i]
                # cache[ids[i]] = names[i]
                pass
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"Train finished:{img_folder} at:{now}\n")
        except Exception as e:
            logging.error(e)
            return "Error with {}".format(e)
    print('All train finished at:{}\n\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    create_index(index_client, table_name)
    print('Indexing finished at:{}\n\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    return "Train finished"

