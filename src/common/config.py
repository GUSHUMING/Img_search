import os

MILVUS_HOST =  "192.168.1.178"
MILVUS_PORT =  19530
VECTOR_DIMENSION = 512
DATA_PATH = "/mnt/weeddata/imgs"
DEFAULT_TABLE =  "imgs_all"
UPLOAD_PATH = "/home/gushuming/upload_data"

default_indexer = "milvus"
default_cache_dir = "/home/gushuming/cache_new"
input_shape = (224,224,3)