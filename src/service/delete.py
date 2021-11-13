import logging

from common.config import DEFAULT_TABLE
from indexer.index import milvus_client, delete_table


def do_delete(table_name):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        index_client = milvus_client()
        status = delete_table(index_client, table_name=table_name)
        return status
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e)

