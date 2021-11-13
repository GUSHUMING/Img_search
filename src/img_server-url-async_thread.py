import argparse
import json
import os
import re
import time

import tornado.gen
import tornado.httpclient
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from service.search import do_search
from torch_model.vgg16 import vgg16_net

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ALLOWED_EXTENSIONS = set(['jpg', 'png'])
# app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
# app.config['JSON_SORT_KEYS'] = False

model = vgg16_net()


def do_search_api(file_path):
    table_name = 'imgs_all'
    top_k = 10
    time_search = time.time()

    # return time_search
    res_id, res_distance = do_search(table_name, file_path, top_k, model)

    print(f'do_search:{time_search}')
    res_img = res_id
    # print('res_img:', res_img)
    res = dict(zip(res_img, res_distance))

    ## score filter
    res = {k: v for k, v in res.items() if res[k] <= 0.45}
    res = sorted(res.items(), key=lambda item: item[1])
    return res


# 命令行参数解析
def parse_args():
    # 使用argparse对命令行参数进行解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--operation',
        default=4,
        type=int,
        help='type of operations,1 for train,2 for train continue,3 for predict.')
    parser.add_argument(
        '--server',
        type=str,
        default='',
        help='server')
    # 返回解析结果
    return parser.parse_known_args()


class ApiHanlder(tornado.web.RequestHandler):
    async def get(self):
        url = self.get_query_argument('url', '')  # 如果没有参数时，返回空字符串
        print("1111",url)
        time1 = time.time()
        ret = await self.getSim(url)
        time2 = time.time()
        filename = re.search(r'([-_\w]+\.(?:jpg|jpeg|png))', url, re.I).group(1)
        file_path = os.path.join(download_path, filename)

        with open(file_path, "wb") as f:
            f.write(ret)
        time3 = time.time()
        print(f'file_path:{file_path}\nbegin:{time1}\ndownload:{time2}\nstore:{time3}')
        # ret = do_search_api(file_path)
        # self.write(json.dumps(ret))

        ret = await self.call_blocking(file_path)
        print(f'ret:{url} => {file_path} => {ret}')
        self.write(json.dumps(ret))
        self.set_header('Access-Control-Allow-Origin', '*')

    async def getSim(self, url):
        print(f'url:{url}')
        # urlretrieve(url, file_path)
        try:
            response = await tornado.httpclient.AsyncHTTPClient().fetch(url, request_timeout=5)
        except Exception as e:
            print("Error: %s" % e)
        else:
            # print(response.body)
            pass

        time2 = time.time()
        # ret = do_search_api(file_path)
        time3 = time.time()
        # print(f'upload:{time2}\nfinish:{time3}\n')
        return response.body

    async def call_blocking(self, file_path):
        ret = await tornado.ioloop.IOLoop.current().run_in_executor(None, getSearch, file_path)
        return ret
# download_path = os.path.join(os.path.dirname(__file__), '../download_img')

download_path = '/home/gushuming/imgSimServer/download_img'
handlers = [
    (r"/file", ApiHanlder),
]


def model_server(addr):
    if ":" in addr:
        host, port = addr.split(":")
    else:
        host = "192.168.1.179"
        port = 8085
    print(f"bind: {addr}:{port}")
    app = tornado.web.Application(handlers, debug=True)
    print("app")
    http_server = tornado.httpserver.HTTPServer(app)
    print("http_server = tornado.httpserver.HTTPServer(app)")
    http_server.listen(int(port))
    print('http_server.listen(int(port))')
    http_server.start(num_processes=1)  # 根据CPU核数fork工作进程个数
    print('http_server.start(num_processes=1)')
    tornado.ioloop.IOLoop.instance().start()


def getSearch(file_path):
    ret = do_search_api(file_path)
    # print(f'searchret:{ret}')
    return ret


if __name__ == '__main__':
    # 命令行参数
    # FLAGS, unparsed = parse_args()
    # print("FLAGS.server:",FLAGS.server)
    # if FLAGS.server:
    #
    #     print('11')
    model_server("192.168.1.179")
        # exit(0)
