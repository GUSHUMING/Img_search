#import imghdr
# import imghdr
import io
import os

from PIL import Image

from encoder.utils import get_imlist


# def is_valid_img(img_path):
#     try:
#         img_type = imghdr.what(img_path)
#     except:
#         return 0
#     if img_type == 'jpeg' or img_type == 'png':
#         return 1
#     return 0


def IsValidImage(pathfile):
    '''
    # 判断文件是否为有效（完整）的图片
    # 输入参数为文件路径
    '''
    bValid = True
    try:
        Image.open(pathfile).verify()
    except:
        bValid = False
    return bValid

def IsValidImage4Bytes(buf):
    '''
    # 判断文件是否为有效（完整）的图片
    # 输入参数为bytes，如网络请求返回的二进制数据
    '''
    bValid = True
    try:
        Image.open(io.BytesIO(buf)).verify()
    except:
        bValid = False
    return bValid

def feature_extract(database_path, model):
#    cache = Cache(default_cache_dir)
    feats = []
    names = []
    img_list = get_imlist(database_path)
    model = model
    for i, img_path in enumerate(img_list):
        # if i % 5000 == 0:
        #     now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #     print(f'{database_path} cnt: {i} and now:{now}', flush=True)
        # is_valid_img_tag = IsValidImage(img_path)
        # if not is_valid_img_tag:
        #     print(f"invalid img file:{img_path}")
        #     continue
        try:
            norm_feat = model.vgg_extract_feat(img_path)
        except:
            print(f"invalid img file:{img_path}")
            continue
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        current = i+1
        total = len(img_list)
        # cache['current'] = current
        # cache['total'] = total
        #print ("extracting feature from image No. %d , %d images in total" %(current, total))
#    feats = np.array(feats)
    return feats, names
